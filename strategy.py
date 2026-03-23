#!/usr/bin/env python3
"""
Experiment #700: 1h Primary + 4h/12h HTF — Regime-Adaptive RSI + HMA Trend + Volume

Hypothesis: Lower TF (1h) strategies fail due to either (a) too many filters = 0 trades,
or (b) too few filters = fee drag from overtrading. This strategy uses ADX to detect
regime (trending vs ranging) and applies DIFFERENT entry logic per regime:
- Trending (ADX>25): RSI pullback entries in HTF trend direction only
- Ranging (ADX<20): RSI extreme mean-reversion at BB bounds

Key innovations vs failed 1h strategies (#690, #695, #698):
1. LOOSER RSI thresholds (35/65 not 25/75) to ensure trade frequency
2. Volume filter is SOFT (0.6x avg, not 1.2x) - just avoid dead sessions
3. Session filter ONLY affects entry timing, not exit/hold logic
4. 4h HMA for trend bias (proven in #689, #691) - simpler than 1d+1w combo
5. ATR-based position sizing reduction in high vol (protects from 2022 crash)

Why this should work on ALL symbols:
- RSI pullback in trend works on BTC/ETH (not just SOL)
- Mean-reversion in range captures 2025 bear/range market
- 4h HTF filter reduces 1h noise without killing trade frequency
- Target: 40-70 trades/year on 1h (within fee drag limits)

Position sizing: 0.25 base (conservative for 1h), reduced to 0.15 in high vol
Stoploss: 2.5x ATR trailing (mandatory per rules)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_hma_volume_4h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index - standard Wilder formula."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, atr

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    half = period // 2
    wma1 = pd.Series(series).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = pd.Series(hma_raw).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    return hma

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands for mean-reversion levels."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, pct_b

def calculate_volume_avg(volume, period=20):
    """Rolling average volume for volume filter."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    adx_1h, atr_1h = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
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
        if np.isnan(rsi_1h[i]) or np.isnan(adx_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(pct_b[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(vol_avg[i]):
            continue
        if atr_1h[i] <= 1e-10 or vol_avg[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (ADX) ===
        is_trending = adx_1h[i] > 25
        is_ranging = adx_1h[i] < 20
        
        # === TREND BIAS (4h HMA) ===
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER (SOFT - just avoid dead sessions) ===
        volume_ok = volume[i] > 0.6 * vol_avg[i]
        
        # === SESSION FILTER (UTC 8-20 for better liquidity) ===
        # Extract hour from open_time (Binance timestamps are ms since epoch)
        open_time_ms = prices["open_time"].iloc[i]
        hour_utc = (open_time_ms // 3600000) % 24
        session_ok = 8 <= hour_utc <= 20
        
        # === ATR-BASED SIZE ADJUSTMENT ===
        atr_median = np.nanmedian(atr_1h[max(0, i-100):i+1])
        atr_ratio = atr_1h[i] / (atr_median + 1e-10)
        current_size = REDUCED_SIZE if atr_ratio > 1.5 else BASE_SIZE
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25) ===
        if is_trending:
            # Long: Pullback in uptrend (RSI 35-50) + 4h bullish + volume ok
            if trend_bullish and 35 <= rsi_1h[i] <= 50 and volume_ok:
                desired_signal = current_size
            
            # Short: Pullback in downtrend (RSI 50-65) + 4h bearish + volume ok
            elif trend_bearish and 50 <= rsi_1h[i] <= 65 and volume_ok:
                desired_signal = -current_size
            
            # Strong breakout: RSI crosses 50 with momentum
            elif trend_bullish and rsi_1h[i] > 55 and volume_ok and session_ok:
                desired_signal = current_size * 0.8
            elif trend_bearish and rsi_1h[i] < 45 and volume_ok and session_ok:
                desired_signal = -current_size * 0.8
        
        # === RANGING REGIME (ADX < 20) ===
        elif is_ranging:
            # Long: RSI oversold + near BB lower + 4h not strongly bearish
            if rsi_1h[i] < 35 and pct_b[i] < 0.2 and not trend_bearish:
                desired_signal = current_size
            
            # Short: RSI overbought + near BB upper + 4h not strongly bullish
            elif rsi_1h[i] > 65 and pct_b[i] > 0.8 and not trend_bullish:
                desired_signal = -current_size
            
            # Extreme mean-reversion (ignore 4h trend)
            elif rsi_1h[i] < 25 and pct_b[i] < 0.1:
                desired_signal = current_size * 0.6
            elif rsi_1h[i] > 75 and pct_b[i] > 0.9:
                desired_signal = -current_size * 0.6
        
        # === NEUTRAL REGIME (ADX 20-25) ===
        else:
            # Only take strong signals in neutral regime
            if rsi_1h[i] < 30 and trend_bullish and volume_ok:
                desired_signal = current_size * 0.5
            elif rsi_1h[i] > 70 and trend_bearish and volume_ok:
                desired_signal = -current_size * 0.5
        
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
                # Hold long if 4h trend still bullish and RSI not overbought
                if trend_bullish and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_bearish and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: RSI overbought OR 4h trend reverses bearish
        if in_position and position_side > 0:
            if rsi_1h[i] > 75:
                desired_signal = 0.0
            elif close[i] < hma_4h_aligned[i] and adx_1h[i] > 25:
                desired_signal = 0.0
        
        # Short exit: RSI oversold OR 4h trend reverses bullish
        if in_position and position_side < 0:
            if rsi_1h[i] < 25:
                desired_signal = 0.0
            elif close[i] > hma_4h_aligned[i] and adx_1h[i] > 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
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