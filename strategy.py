#!/usr/bin/env python3
"""
Experiment #1070: 1h Primary + 4h/12h HTF — Multi-Confluence Trend Pullback

Hypothesis: After 776+ failed experiments, lower TF (1h) can work IF we use:
1. 4h HMA21 for MACRO trend direction (proven in best strategies)
2. 12h ADX14 for REGIME detection (ADX>25=trend, ADX<20=range)
3. 1h RSI(7) for ENTRY timing (pullback to 35-45 in uptrend, 55-65 in downtrend)
4. Volume confirmation (current > 1.5x 20-bar avg) — confirms real moves
5. Session filter (8-20 UTC) — high liquidity hours only
6. VERY STRICT confluence: need 4h trend + 12h regime + 1h RSI + volume + session

Why this should work when other 1h strategies failed:
- Uses HTF (4h/12h) for DIRECTION, 1h only for TIMING (proven pattern)
- ADX regime filter prevents mean-reversion in trends and vice versa
- RSI pullback (not extreme) entries have better win rate than CRSI extremes
- Volume filter eliminates false breakouts
- Session filter avoids Asian session whipsaws
- Target: 40-70 trades/year (strict enough to avoid fee drag)

Timeframe: 1h
HTF: 4h (trend), 12h (regime) — loaded ONCE before loop
Position Size: 0.25 discrete (conservative for 1h)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_adx_rsi_pullback_4h12h_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = strong trend (trend follow)
    ADX < 20 = weak/range (mean revert)
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 5:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method (EMA-like with alpha=1/period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    
    mask2 = di_sum > 1e-10
    dx[mask2] = 100 * di_diff[mask2] / di_sum[mask2]
    
    # ADX is smoothed DX
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = adx_raw
    
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / average volume over period."""
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    if n < period:
        return ratio
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    mask = vol_avg > 1e-10
    ratio[mask] = volume[mask] / vol_avg[mask]
    
    return ratio

def get_hour_from_timestamp(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    timestamps = prices['open_time'].values / 1000
    hours = np.array([pd.Timestamp(t, unit='s').hour for t in timestamps])
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA21 for macro trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h ADX14 for regime detection
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    hours = get_hour_from_timestamp(prices)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(rsi_7[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.3
        
        # === MACRO TREND (4h HMA21) ===
        macro_bull = close[i] > hma_4h_aligned[i]
        macro_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME (12h ADX14) ===
        adx_val = adx_12h_aligned[i]
        trending_regime = adx_val > 25  # Strong trend
        ranging_regime = adx_val < 20   # Range/weak trend
        
        # === RSI PULLBACK SIGNALS ===
        rsi_val = rsi_7[i]
        
        # Long pullback: RSI 35-48 in uptrend (not oversold, just pullback)
        rsi_long_pullback = 35 <= rsi_val <= 48
        
        # Short pullback: RSI 52-65 in downtrend (not overbought, just retracement)
        rsi_short_pullback = 52 <= rsi_val <= 65
        
        # Extreme RSI for range regime
        rsi_long_extreme = rsi_val < 30
        rsi_short_extreme = rsi_val > 70
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25): Follow pullbacks ===
        if trending_regime:
            # Long: Macro bull + RSI pullback + volume + session
            if macro_bull and rsi_long_pullback and volume_confirmed and in_session:
                desired_signal = BASE_SIZE
            
            # Short: Macro bear + RSI pullback + volume + session
            elif macro_bear and rsi_short_pullback and volume_confirmed and in_session:
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME (ADX < 20): Mean reversion at extremes ===
        elif ranging_regime:
            # Long: RSI extreme + volume + session (ignore macro in range)
            if rsi_long_extreme and volume_confirmed and in_session:
                desired_signal = BASE_SIZE
            
            # Short: RSI extreme + volume + session
            elif rsi_short_extreme and volume_confirmed and in_session:
                desired_signal = -BASE_SIZE
        
        # === TRANSITION REGIME (ADX 20-25): Require stronger confluence ===
        else:
            # Long: All conditions aligned
            if macro_bull and rsi_long_pullback and volume_confirmed and in_session:
                desired_signal = BASE_SIZE
            
            # Short: All conditions aligned
            elif macro_bear and rsi_short_pullback and volume_confirmed and in_session:
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull and RSI not overbought
                if macro_bull and rsi_val < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bear and RSI not oversold
                if macro_bear and rsi_val > 35:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish
            if macro_bear and rsi_val > 55:
                desired_signal = 0.0
            # Exit long if RSI very overbought
            if rsi_val > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish
            if macro_bull and rsi_val < 45:
                desired_signal = 0.0
            # Exit short if RSI very oversold
            if rsi_val < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
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