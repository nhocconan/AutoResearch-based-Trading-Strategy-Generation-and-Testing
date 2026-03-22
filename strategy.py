#!/usr/bin/env python3
"""
Experiment #027: 1d KAMA Adaptive Trend + 1w HMA Bias + Fisher Transform Entries

Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to volatility better than EMA/HMA,
reducing whipsaw in choppy markets (2022 crash, 2025 bear). Combined with:
1. 1w HMA(21) for major regime bias (bull/bear) - call ONCE before loop via mtf_data
2. 1d KAMA(10) for adaptive trend following (ER-based smoothing)
3. Fisher Transform(9) for reversal entry timing (catches bear market rallies)
4. ATR(14) for stoploss (2.5x) and volatility filter
5. Asymmetric positioning: only short in bear regime, only long in bull regime
6. Very strict entry to ensure 15-25 trades/year (optimal for 1d)

Why this should work:
- KAMA adapts smoothing based on volatility ratio (Efficiency Ratio)
- Fisher Transform normalizes price to Gaussian distribution for cleaner signals
- 1w HTF prevents counter-trend trades (major failure mode in 2022)
- Asymmetric regime reduces whipsaw (don't long in bear, don't short in bull)
- 1d timeframe targets 15-25 trades/year (minimal fee drag)
- Discrete sizing 0.25-0.30 with ATR stoploss

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 strong trend
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_1w_hma_asymmetric_v1"
timeframe = "1d"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    ER = |net change| / sum of absolute changes over period
    SC = (ER * (fast_SC - slow_SC) + slow_SC)^2
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio calculation
    for i in range(period, n):
        # Net change over period
        net_change = np.abs(close[i] - close[i - period])
        
        # Sum of absolute changes (volatility/noise)
        sum_changes = 0.0
        for j in range(1, period + 1):
            sum_changes += np.abs(close[i - j + 1] - close[i - j])
        
        # Efficiency Ratio (0 = noise, 1 = perfect trend)
        if sum_changes > 0:
            er = net_change / sum_changes
        else:
            er = 0.0
        
        # Smoothing Constant
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    # Fill initial values
    kama[:period] = close[:period]
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for cleaner reversal signals.
    Entry: Fisher crosses above -1.5 (long), below +1.5 (short)
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        # Normalize price to 0-1 range
        if hh != ll:
            value = 0.6667 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * (
                0.6667 * ((close[i - 1] - ll) / (hh - ll) - 0.5) + 0.67 * (
                    0.6667 * ((close[i - 2] - ll) / (hh - ll) - 0.5)
                    if i >= 2 else 0
                )
            )
        else:
            value = 0.0
        
        # Clamp to avoid division issues
        value = np.clip(value, -0.999, 0.999)
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        # Signal line (previous fisher)
        if i > period:
            fisher_signal[i] = fisher[i - 1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
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
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_s / np.where(tr_s == 0, 1e-10, tr_s)
    minus_di = 100 * minus_dm_s / np.where(tr_s == 0, 1e-10, tr_s)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, 1e-10, plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA trend for regime bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_10 = calculate_kama(close, period=10)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Also calculate 1d HMA for additional confirmation
    hma_1d_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(kama_10[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        # === 1W HTF REGIME BIAS (ASYMMETRIC) ===
        # Bull regime: price above 1w HMA (only take longs)
        # Bear regime: price below 1w HMA (only take shorts)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        local_bullish = close[i] > kama_10[i]
        local_bearish = close[i] < kama_10[i]
        
        # KAMA slope confirmation
        kama_slope_bullish = kama_10[i] > kama_10[i - 5] if i >= 5 else False
        kama_slope_bearish = kama_10[i] < kama_10[i - 5] if i >= 5 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20
        adx_weak = adx_14[i] <= 20
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Alternative: Fisher extreme reversals
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if bull_regime and local_bullish and adx_strong:
            current_size = STRONG_SIZE
        elif bull_regime and local_bullish:
            current_size = BASE_SIZE
        elif bear_regime and local_bearish and adx_strong:
            current_size = STRONG_SIZE
        elif bear_regime and local_bearish:
            current_size = BASE_SIZE
        else:
            current_size = BASE_SIZE * 0.8
        
        # === ENTRY LOGIC (ASYMMETRIC REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + KAMA bullish + Fisher reversal
        if bull_regime and local_bullish and kama_slope_bullish:
            if fisher_long or (fisher_oversold and local_bullish):
                new_signal = current_size
        
        # SHORT ENTRY: Bear regime + KAMA bearish + Fisher reversal
        if bear_regime and local_bearish and kama_slope_bearish:
            if fisher_short or (fisher_overbought and local_bearish):
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 45 bars (~45 days on 1d), allow weaker entry
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if bull_regime and local_bullish and fisher_oversold:
                new_signal = current_size * 0.8
            elif bear_regime and local_bearish and fisher_overbought:
                new_signal = -current_size * 0.8
        
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if regime changes against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if regime turns bear
            if position_side > 0 and bear_regime:
                regime_reversal = True
            # Exit short if regime turns bull
            if position_side < 0 and bull_regime:
                regime_reversal = True
        
        # === KAMA REVERSAL EXIT ===
        kama_reversal = False
        if in_position and position_side != 0:
            # Exit long if KAMA turns bearish
            if position_side > 0 and local_bearish and kama_slope_bearish:
                kama_reversal = True
            # Exit short if KAMA turns bullish
            if position_side < 0 and local_bullish and kama_slope_bullish:
                kama_reversal = True
        
        # === FISHER EXTREME EXIT (take profit) ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher becomes very overbought
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            # Exit short when Fisher becomes very oversold
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_reversal or kama_reversal or fisher_exit:
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