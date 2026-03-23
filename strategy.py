#!/usr/bin/env python3
"""
Experiment #703: 1d Primary + 1w HTF — Ehlers Fisher Transform + HMA Trend + Volume

Hypothesis: Ehlers Fisher Transform (period=9) excels at identifying turning points
in daily data by normalizing price into a Gaussian distribution. Combined with
1w HMA for major trend bias and volume confirmation for breakouts, this captures
reversals in bear markets while riding trends in bull markets.

Why this should work:
1. Fisher Transform is proven for daily TF (John Ehlers' research)
2. 1d timeframe worked in past experiments (lower noise than 4h/12h)
3. 1w HMA provides strong trend bias without over-filtering
4. Volume confirmation filters false breakouts
5. Simple RSI(14) extremes for mean-reversion entries
6. ATR-based stops prevent catastrophic drawdowns

Key differences from failed experiments:
- Fisher Transform instead of CRSI (different signal source)
- Volume ratio confirmation (not used in recent failures)
- Simpler entry logic (fewer confluence = more trades)
- Asymmetric sizing (larger in trend, smaller in mean-revert)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_hma_volume_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price into Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period:
        return fisher, trigger
    
    # Calculate typical price
    typical = (high + low) / 2
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price to range -1 to +1
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            trigger[i] = fisher[i]
            continue
        
        normalized = 0.66 * ((typical[i] - lowest) / range_val - 0.5) + 0.67 * (fisher[i-1] if i > 0 else 0.0)
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs recent average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_avg + 1e-10)
    return ratio

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    TREND_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_ratio[i]) or atr_1d[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1w HMA) ===
        trend_bullish = close[i] > hma_1w_aligned[i]
        trend_bearish = close[i] < hma_1w_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_1d[i] < 35
        rsi_overbought = rsi_1d[i] > 65
        
        # === ENTRY LOGIC ===
        
        # TREND FOLLOWING (with trend bias + volume)
        if trend_bullish and above_sma200:
            # Long on Fisher reversal or RSI oversold with volume
            if fisher_long_cross and volume_confirmed:
                desired_signal = TREND_SIZE
            elif rsi_oversold and volume_confirmed:
                desired_signal = current_size
        
        elif trend_bearish and below_sma200:
            # Short on Fisher reversal or RSI overbought with volume
            if fisher_short_cross and volume_confirmed:
                desired_signal = -TREND_SIZE
            elif rsi_overbought and volume_confirmed:
                desired_signal = -current_size
        
        # MEAN REVERSION (counter-trend in ranges)
        else:
            # Long: RSI very oversold (<25) without volume requirement
            if rsi_1d[i] < 25:
                desired_signal = REDUCED_SIZE
            # Short: RSI very overbought (>75) without volume requirement
            elif rsi_1d[i] > 75:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish and RSI not extreme
                if trend_bullish and rsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish and RSI not extreme
                if trend_bearish and rsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: RSI overbought OR trend reverses bearish
        if in_position and position_side > 0:
            if rsi_1d[i] > 80:
                desired_signal = 0.0
            elif close[i] < hma_1w_aligned[i] and rsi_1d[i] > 50:
                desired_signal = 0.0
        
        # Short exit: RSI oversold OR trend reverses bullish
        if in_position and position_side < 0:
            if rsi_1d[i] < 20:
                desired_signal = 0.0
            elif close[i] > hma_1w_aligned[i] and rsi_1d[i] < 50:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= TREND_SIZE * 0.9:
                desired_signal = TREND_SIZE
            elif desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -TREND_SIZE * 0.9:
                desired_signal = -TREND_SIZE
            elif desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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