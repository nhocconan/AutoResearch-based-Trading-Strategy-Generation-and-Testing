#!/usr/bin/env python3
"""
Experiment #1050: 1h Primary + 4h/12h HTF — Regime-Adaptive Mean Reversion with HTF Trend Bias

Hypothesis: After analyzing 760+ failed experiments, the key insight is:
1. Lower TF (1h) strategies fail because entry conditions are TOO STRICT (0 trades)
2. The winning approach: Use 4h/12h for TREND DIRECTION, 1h only for ENTRY TIMING
3. This gives HTF trade frequency (30-60/year) with lower TF execution precision

Strategy Components:
1. 4h HMA21: Macro trend filter (only long when price > 4h_HMA, only short when <)
2. 12h HMA21: Secondary confirmation (strengthens bias)
3. Choppiness Index (14): Regime detection (>55 = range/mean-revert, <45 = trend/follow)
4. RSI(7): Faster RSI for 1h entries (oversold <35, overbought >65)
5. Bollinger Bands (20, 2.0): Mean reversion levels in range regime
6. Volume filter: >0.7x 20-bar average (not too strict)
7. Session filter: 8-20 UTC only (London/NY overlap - highest liquidity)
8. ATR trailing stop: 2.5x for risk management

Why this should work:
- 4h HMA provides macro bias without being too restrictive (unlike 1d)
- 1h RSI(7) is faster than RSI(14) for entries (more signals)
- Choppiness regime filter adapts to market conditions
- Relaxed volume (0.7x not 1.0x) and RSI (35/65 not 30/70) ensures trades
- Session filter reduces noise but doesn't eliminate all opportunities

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year)
Position Size: 0.25 discrete levels (smaller for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_4h12h_hma_session_vol_atr_v2"
timeframe = "1h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=7):
    """Fast RSI for 1h entries."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stops."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average - faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion entries."""
    close_series = pd.Series(close)
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    middle = rolling_mean.values
    upper = (rolling_mean + std_mult * rolling_std).values
    lower = (rolling_mean - std_mult * rolling_std).values
    
    return upper, middle, lower

def calculate_volume_ma(volume, period=20):
    """Volume moving average for volume filter."""
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA21 for macro trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for more signals
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 1h TF
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
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_ma[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Ranging market (mean reversion)
        is_trend = chop[i] < 45.0  # Trending market (trend following)
        
        # === MACRO TREND (4h HMA21 + 12h HMA21) ===
        macro_bull_4h = close[i] > hma_4h_aligned[i]
        macro_bear_4h = close[i] < hma_4h_aligned[i]
        macro_bull_12h = close[i] > hma_12h_aligned[i]
        macro_bear_12h = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both agree
        macro_bull = macro_bull_4h and macro_bull_12h
        macro_bear = macro_bear_4h and macro_bear_12h
        macro_neutral = not macro_bull and not macro_bear
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION ===
        if is_range and in_session and volume_ok:
            # Long: RSI oversold + price at/near BB lower + macro bullish or neutral
            if rsi[i] < 35 and close[i] <= bb_lower[i] * 1.002:  # Small buffer
                if macro_bull or macro_neutral:
                    desired_signal = BASE_SIZE
            # Short: RSI overbought + price at/near BB upper + macro bearish or neutral
            elif rsi[i] > 65 and close[i] >= bb_upper[i] * 0.998:
                if macro_bear or macro_neutral:
                    desired_signal = -BASE_SIZE
            # Weaker signals (relaxed RSI)
            elif rsi[i] < 30:
                if macro_bull or macro_neutral:
                    desired_signal = REDUCED_SIZE
            elif rsi[i] > 70:
                if macro_bear or macro_neutral:
                    desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: TREND FOLLOWING ===
        elif is_trend and in_session and volume_ok:
            # Long: pullback in uptrend (RSI 40-50) + macro bullish
            if macro_bull and 40 <= rsi[i] <= 55:
                desired_signal = BASE_SIZE
            # Short: pullback in downtrend (RSI 45-60) + macro bearish
            elif macro_bear and 45 <= rsi[i] <= 60:
                desired_signal = -BASE_SIZE
            # Strong trend continuation
            elif macro_bull and rsi[i] > 55 and rsi[i] < 70:
                desired_signal = REDUCED_SIZE
            elif macro_bear and rsi[i] < 45 and rsi[i] > 30:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45-55 CHOP) ===
        else:
            # Only take strongest signals in neutral regime
            if in_session and volume_ok:
                if rsi[i] < 25 and (macro_bull or macro_neutral):
                    desired_signal = REDUCED_SIZE
                elif rsi[i] > 75 and (macro_bear or macro_neutral):
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish or RSI not overbought
                if (macro_bull or macro_neutral) and rsi[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish or RSI not oversold
                if (macro_bear or macro_neutral) and rsi[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses strongly bearish
            if macro_bear and rsi[i] > 55:
                desired_signal = 0.0
            # Exit long if RSI very overbought
            if rsi[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses strongly bullish
            if macro_bull and rsi[i] < 45:
                desired_signal = 0.0
            # Exit short if RSI very oversold
            if rsi[i] < 25:
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
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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