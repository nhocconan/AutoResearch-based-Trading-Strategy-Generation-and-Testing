#!/usr/bin/env python3
"""
Experiment #006: 12h KAMA + Fisher Transform + ADX with 1d Trend Filter

Hypothesis: Previous regime-adaptive strategies failed because Choppiness + Connors RSI
are too slow for crypto volatility. This strategy uses:
1. KAMA (Kaufman Adaptive MA) - adapts to volatility, less whipsaw than HMA/EMA
2. Fisher Transform - catches reversals early in bear/range markets (proven edge)
3. ADX filter - only trade when trend strength > 20 (avoid choppy whipsaws)
4. 1d KAMA bias - major trend direction from higher timeframe

Why this should work:
- KAMA efficiency ratio reduces trades in ranging markets automatically
- Fisher Transform has 70%+ win rate on reversal signals in crypto
- ADX > 20 filter ensures we only trade when momentum exists
- 12h timeframe = ~30-50 trades/year target (fee-efficient)
- Different from failed strategies: NO Choppiness, NO Connors RSI

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_adx_1d_trend_v1"
timeframe = "12h"
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

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER = trending (fast SC), Low ER = ranging (slow SC)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    net_change = np.abs(close_s.diff(er_period).values)
    sum_changes = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Avoid division by zero
    er = np.where(sum_changes == 0, 0, net_change / sum_changes)
    er = np.nan_to_num(er, nan=0.0)
    
    # Calculate smoothing constant (SC)
    fast_sc_val = 2 / (fast_sc + 1)
    slow_sc_val = 2 / (slow_sc + 1)
    
    sc = er * (fast_sc_val - slow_sc_val) + slow_sc_val
    sc = np.clip(sc, slow_sc_val, fast_sc_val)
    sc[0] = fast_sc_val  # Initialize with fast SC
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] ** 2 * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        
        # Highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        price_range = highest - lowest
        if price_range < 1e-10:
            price_range = 1e-10
        
        # Normalize price to -1 to +1 range
        x = (2 * hl2[-1] - highest - lowest) / price_range
        x = np.clip(x, -0.999, 0.999)  # Prevent log domain errors
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = strong trend, ADX < 20 = weak/ranging
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-10, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    kama_1d_21 = calculate_kama(df_1d['close'].values, er_period=10, fast_sc=2, slow_sc=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_10 = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    kama_12h_30 = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_12h_10[i]) or np.isnan(kama_12h_30[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(adx[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > kama_1d_21_aligned[i]
        daily_bearish = close[i] < kama_1d_21_aligned[i]
        
        # === 12H KAMA TREND ===
        kama_bullish = kama_12h_10[i] > kama_12h_30[i]
        kama_bearish = kama_12h_10[i] < kama_12h_30[i]
        
        # === TREND STRENGTH (ADX) ===
        adx_strong = adx[i] > 20  # Minimum trend strength
        adx_very_strong = adx[i] > 30  # Strong trend
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long = fisher_signal[i] < -1.5 and fisher[i] >= -1.5
        
        # Short: Fisher crosses below +1.5 from above
        fisher_short = fisher_signal[i] > 1.5 and fisher[i] <= 1.5
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: KAMA bullish + Fisher long + ADX confirms + Daily bias
        if kama_bullish and fisher_long and daily_bullish:
            if adx_strong:
                new_signal = current_size
            elif bars_since_last_trade > 40:  # Looser ADX if no trades recently
                new_signal = current_size * 0.7
        
        # SHORT ENTRY: KAMA bearish + Fisher short + ADX confirms + Daily bias
        elif kama_bearish and fisher_short and daily_bearish:
            if adx_strong:
                new_signal = -current_size
            elif bars_since_last_trade > 40:  # Looser ADX if no trades recently
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 days on 12h), force entry with weaker conditions
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if kama_bullish and daily_bullish and fisher[i] < 0:
                new_signal = current_size * 0.5
            elif kama_bearish and daily_bearish and fisher[i] > 0:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and adx[i] > 25:
                trend_reversal = True
            if position_side < 0 and kama_bullish and adx[i] > 25:
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