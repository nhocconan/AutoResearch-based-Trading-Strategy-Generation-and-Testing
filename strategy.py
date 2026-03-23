#!/usr/bin/env python3
"""
Experiment #900: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + Volume

Hypothesis: After 633 failed strategies, 1h timeframe needs SIMPLER entry logic
to guarantee trades while maintaining edge. Key lessons from failures:

1. Session filters kill trade generation on 1h (experiments #890, #895 = 0 trades)
2. Multiple HTF filters + volume + session = too restrictive
3. Fisher Transform catches reversals better than RSI in bear/range markets
4. Need MULTIPLE entry paths to guarantee 30+ trades per symbol
5. Hold logic is critical - maintain position through minor pullbacks

Strategy design:
- 4h HMA(21) for trend direction (proven in baseline)
- 12h HMA(21) for macro regime confirmation
- Fisher Transform(9) for entry timing (more sensitive than RSI)
- Volume filter relaxed to 0.5x avg (not 0.8x)
- NO session filter (killed trades in #890, #895)
- RSI fallback when Fisher doesn't trigger
- 2 regime states (range/trend) not 3

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-70 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_regime_4h12h_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Better than RSI for catching reversals in bear markets.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, trigger
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0
            trigger[i] = 0
            continue
        
        price = (high[i] + low[i]) / 2
        normalized = (price - lowest) / (highest - lowest)
        normalized = np.clip(normalized, 0.001, 0.999)
        
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        fisher[i] = fisher_val
        
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher_val
    
    return fisher, trigger

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Average volume over period."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    fisher_1h, trigger_1h = calculate_fisher_transform(high, low, period=9)
    rsi_1h = calculate_rsi(close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_avg_1h = calculate_volume_avg(volume, period=20)
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for regime confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_1h[i]) or np.isnan(trigger_1h[i]):
            continue
        if np.isnan(rsi_1h[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(vol_avg_1h[i]) or vol_avg_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === VOLUME CONFIRMATION (Relaxed: 0.5x avg) ===
        volume_confirmed = volume[i] > 0.5 * vol_avg_1h[i]
        
        # === TREND DIRECTION (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME CONFIRMATION (12h HMA21) ===
        regime_12h_bullish = close[i] > hma_12h_aligned[i]
        regime_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (1h Choppiness) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(trigger_1h[i-1]):
            # Long: Fisher crosses above -1.5
            fisher_cross_long = fisher_1h[i] > -1.5 and trigger_1h[i-1] <= -1.5
            # Short: Fisher crosses below +1.5
            fisher_cross_short = fisher_1h[i] < 1.5 and trigger_1h[i-1] >= 1.5
        
        # Extreme Fisher levels
        fisher_extreme_long = fisher_1h[i] < -2.0
        fisher_extreme_short = fisher_1h[i] > 2.0
        
        # === RSI SIGNALS (Fallback) ===
        rsi_oversold = rsi_1h[i] < 35
        rsi_overbought = rsi_1h[i] > 65
        rsi_extreme_oversold = rsi_1h[i] < 25
        rsi_extreme_overbought = rsi_1h[i] > 75
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Primary: Fisher extreme + volume
            if fisher_extreme_long and volume_confirmed:
                desired_signal = BASE_SIZE
            
            if fisher_extreme_short and volume_confirmed:
                desired_signal = -BASE_SIZE
            
            # Secondary: Fisher crossover + trend alignment
            if fisher_cross_long and (trend_4h_bullish or regime_12h_bullish) and volume_confirmed:
                if desired_signal == 0:
                    desired_signal = REDUCED_SIZE
            
            if fisher_cross_short and (trend_4h_bearish or regime_12h_bearish) and volume_confirmed:
                if desired_signal == 0:
                    desired_signal = -REDUCED_SIZE
            
            # Tertiary: RSI extreme (guarantees trades)
            if rsi_extreme_oversold and volume_confirmed and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and volume_confirmed and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + Fisher signal
            if trend_4h_bullish or regime_12h_bullish:
                if fisher_cross_long and volume_confirmed:
                    desired_signal = BASE_SIZE
                elif fisher_1h[i] > -1.0 and volume_confirmed and desired_signal == 0:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + Fisher signal
            if trend_4h_bearish or regime_12h_bearish:
                if fisher_cross_short and volume_confirmed:
                    desired_signal = -BASE_SIZE
                elif fisher_1h[i] < 1.0 and volume_confirmed and desired_signal == 0:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require trend alignment + signal
            if fisher_cross_long and trend_4h_bullish and volume_confirmed:
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_short and trend_4h_bearish and volume_confirmed:
                desired_signal = -REDUCED_SIZE
            
            # Fallback: RSI + HTF alignment
            if rsi_extreme_oversold and trend_4h_bullish and volume_confirmed and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and trend_4h_bearish and volume_confirmed and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish and Fisher not overbought
                if trend_4h_bullish and fisher_1h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and Fisher not oversold
                if trend_4h_bearish and fisher_1h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses + Fisher overbought
            if trend_4h_bearish and fisher_1h[i] > 1.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses + Fisher oversold
            if trend_4h_bullish and fisher_1h[i] < -1.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals