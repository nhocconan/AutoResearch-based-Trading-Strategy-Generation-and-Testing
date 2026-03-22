#!/usr/bin/env python3
"""
Experiment #014: 4h KAMA + Fisher Transform + Volume Confirmation with 12h/1d HTF

Hypothesis: Previous regime-adaptive strategies failed because Choppiness Index
is too laggy and Connors RSI doesn't work well on crypto's persistent trends.

This strategy uses:
1. KAMA (Kaufman Adaptive MA) - adapts speed based on market efficiency ratio
   Faster in trends, slower in chop. Proven to reduce whipsaw vs HMA/EMA.
2. Fisher Transform - normalizes price to Gaussian distribution, catches reversals
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
3. Volume spike confirmation - breakouts need 1.5x avg volume to be valid
4. 12h HMA for intermediate trend bias
5. 1d HMA for major trend bias (only trade with 1d trend)
6. ATR trailing stoploss at 2.0x (tighter than previous 2.5x)

Why this should work:
- KAMA adapts to volatility regimes automatically (no Choppiness needed)
- Fisher Transform catches reversals in bear markets (2025+ is bearish)
- Volume filter reduces false breakouts (major issue in crypto)
- 4h timeframe balances trade frequency (30-60/year) vs signal quality
- HTF bias ensures we don't fight the major trend

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_volume_12h1d_bias_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts speed based on market efficiency ratio.
    ER = |Net Change| / Sum of individual changes (0 to 1)
    SC = (ER * (fast_SC - slow_SC) + slow_SC)^2
    KAMA = KAMA_prev + SC * (price - KAMA_prev)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Net change over period
    net_change = np.abs(close_s - close_s.shift(period)).values
    
    # Sum of individual changes (volatility)
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
    
    # Efficiency Ratio (avoid division by zero)
    er = np.zeros(n)
    for i in range(period, n):
        if volatility[i] > 0:
            er[i] = net_change[i] / volatility[i]
        else:
            er[i] = 0
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]  # Initialize with price
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:period] = close[:period]
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate median price
    median_price = (high + low) / 2
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price to -1 to +1 range
        if hh != ll:
            x = (2 * median_price[i] - hh - ll) / (hh - ll)
        else:
            x = 0
        
        # Clip to avoid log errors
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Trigger line (previous fisher value)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes (volume > threshold * avg volume)."""
    vol_s = pd.Series(volume)
    avg_vol = vol_s.rolling(window=period, min_periods=period).mean().values
    is_spike = volume > (threshold * avg_vol)
    return is_spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_10 = calculate_kama(close, period=10, fast=2, slow=30)
    kama_30 = calculate_kama(close, period=30, fast=2, slow=30)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    volume_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # Additional filters
    close_s = pd.Series(close)
    sma_200 = close_s.rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):  # Start after SMA200 is ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === HTF TREND BIAS ===
        # Only trade with 1d trend direction
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # 12h intermediate trend
        hma_12h_bullish = close[i] > hma_12h_21_aligned[i]
        hma_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # === VOLUME CONFIRMATION ===
        # Breakouts need volume spike to be valid
        vol_confirmed = volume_spike[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Multiple confluence required
        long_conditions = (
            daily_bullish and  # 1d trend up
            hma_12h_bullish and  # 12h trend up
            kama_bullish and  # 4h KAMA bullish
            close[i] > sma_200[i] and  # Above long-term average
            (fisher_long or (kama_10[i] > kama_30[i] and close[i] > kama_10[i] * 0.995))  # Fisher or KAMA pullback
        )
        
        # SHORT ENTRY: Multiple confluence required
        short_conditions = (
            daily_bearish and  # 1d trend down
            hma_12h_bearish and  # 12h trend down
            kama_bearish and  # 4h KAMA bearish
            close[i] < sma_200[i] and  # Below long-term average
            (fisher_short or (kama_10[i] < kama_30[i] and close[i] < kama_10[i] * 1.005))  # Fisher or KAMA pullback
        )
        
        # Volume confirmation for breakouts (relax for pullback entries)
        if long_conditions:
            if fisher_long or vol_confirmed:
                new_signal = current_size
        
        if short_conditions:
            if fisher_short or vol_confirmed:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~6-7 days on 4h), allow weaker entry
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if kama_bullish and daily_bullish and fisher[i] < 0:
                new_signal = current_size * 0.6
            elif kama_bearish and daily_bearish and fisher[i] > 0:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and daily_bearish:
                trend_reversal = True
            if position_side < 0 and kama_bullish and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals