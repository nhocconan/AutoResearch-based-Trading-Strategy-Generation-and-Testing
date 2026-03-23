#!/usr/bin/env python3
"""
Experiment #670: 1h Primary + 4h/12h HTF — Fisher Transform + Session + Volume Confluence

Hypothesis: 1h timeframe with strict confluence filters (HTF trend + Fisher + volume + session)
will achieve optimal trade frequency (30-80/year) while maintaining signal quality.
Key innovation: Fisher Transform for entry timing (superior to RSI in bear/range markets)
combined with UTC session filter (8-20) to avoid low-liquidity whipsaws.

Why this should work:
1. 4h HMA for macro trend bias — prevents counter-trend trades
2. 12h ADX for regime detection — trend vs mean-revert
3. Fisher Transform (period=9) for precise entry timing — catches reversals better than RSI
4. Session filter (8-20 UTC) — avoids Asian session low liquidity periods
5. Volume filter (>0.8x 20-bar avg) — confirms breakout validity
6. Trailing ATR stoploss (2.5x) — protects capital in volatile moves
7. Discrete signal values (0.0, ±0.25, ±0.30) — minimizes fee churn from signal changes

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive Sharpe, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_session_vol_4h12h_confluence_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Better at catching reversals in bear/range markets than RSI.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    # Calculate typical price
    typical = (high + low) / 2
    
    # Normalize price to range -1 to +1
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = 2 * (typical - lowest) / (highest - lowest + 1e-10) - 1
    
    # Clamp to avoid division issues
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher_raw = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher_signal_raw = np.roll(fisher_raw, 1)
    fisher_signal_raw[0] = np.nan
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    fisher_signal = pd.Series(fisher_signal_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        adx[period*2-1:] = adx_raw[period*2-1:]
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    if n < period:
        return vol_ratio
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_avg + 1e-10)
    
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

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
    
    # Calculate 1h indicators (primary timeframe)
    fisher_1h, fisher_signal_1h = calculate_fisher_transform(high, low, period=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(fisher_1h[i]) or np.isnan(fisher_signal_1h[i]):
            continue
        if np.isnan(adx_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(vol_ratio_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(adx_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = vol_ratio_1h[i] > 0.8
        
        # === REGIME DETECTION (12h ADX) ===
        adx_12h_value = adx_12h_aligned[i]
        is_trend_regime = adx_12h_value > 25
        is_chop_regime = adx_12h_value < 20
        
        # === HTF TREND BIAS (4h & 12h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both HTF agree
        htf_strong_bullish = htf_4h_bullish and htf_12h_bullish
        htf_strong_bearish = htf_4h_bearish and htf_12h_bearish
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = fisher_1h[i] > -1.5 and fisher_signal_1h[i] <= -1.5
        fisher_cross_short = fisher_1h[i] < 1.5 and fisher_signal_1h[i] >= 1.5
        
        # Fisher extreme levels
        fisher_oversold = fisher_1h[i] < -1.8
        fisher_overbought = fisher_1h[i] > 1.8
        
        # Fisher reversal (crossing zero)
        fisher_zero_cross_long = fisher_1h[i] > 0 and fisher_signal_1h[i] <= 0
        fisher_zero_cross_short = fisher_1h[i] < 0 and fisher_signal_1h[i] >= 0
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (ADX > 25) — Trend Follow ===
        if is_trend_regime:
            # Long: Strong HTF bullish + Fisher confirmation + session + volume
            if htf_strong_bullish and in_session and volume_confirmed:
                if fisher_zero_cross_long or fisher_cross_long:
                    desired_signal = SIZE_LONG
                elif fisher_oversold and fisher_1h[i] > fisher_signal_1h[i]:
                    # Fisher turning up from oversold
                    desired_signal = SIZE_LONG
            
            # Short: Strong HTF bearish + Fisher confirmation + session + volume
            elif htf_strong_bearish and in_session and volume_confirmed:
                if fisher_zero_cross_short or fisher_cross_short:
                    desired_signal = -SIZE_SHORT
                elif fisher_overbought and fisher_1h[i] < fisher_signal_1h[i]:
                    # Fisher turning down from overbought
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 2: CHOPPY (ADX < 20) — Mean Reversion ===
        elif is_chop_regime:
            # Long: Fisher oversold + HTF not strongly bearish + session + volume
            if fisher_oversold and not htf_strong_bearish and in_session and volume_confirmed:
                if fisher_1h[i] > fisher_signal_1h[i]:  # Fisher turning up
                    desired_signal = SIZE_LONG
            
            # Short: Fisher overbought + HTF not strongly bullish + session + volume
            elif fisher_overbought and not htf_strong_bullish and in_session and volume_confirmed:
                if fisher_1h[i] < fisher_signal_1h[i]:  # Fisher turning down
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION (20 <= ADX <= 25) — Mixed ===
        else:
            # Use 4h HMA direction with Fisher filter
            if htf_4h_bullish and in_session and volume_confirmed:
                if fisher_zero_cross_long or (fisher_1h[i] > -1.0 and fisher_1h[i] > fisher_signal_1h[i]):
                    desired_signal = SIZE_LONG
            elif htf_4h_bearish and in_session and volume_confirmed:
                if fisher_zero_cross_short or (fisher_1h[i] < 1.0 and fisher_1h[i] < fisher_signal_1h[i]):
                    desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h HMA still bullish AND Fisher not extremely overbought
                if htf_4h_bullish and fisher_1h[i] < 2.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if 4h HMA still bearish AND Fisher not extremely oversold
                if htf_4h_bearish and fisher_1h[i] > -2.0:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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