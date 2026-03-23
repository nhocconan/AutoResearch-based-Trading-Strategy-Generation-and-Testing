#!/usr/bin/env python3
"""
Experiment #118: 30m Primary + 4h/1d HTF — Vol Spike Mean Reversion with Regime Filter

Hypothesis: Previous strategies failed because they used pure trend-following in bear/range markets.
Research shows VOL SPIKE REVERSION has strong edge: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long.
This captures "vol crush" after panic selling - high win rate mean reversion.

Key innovations:
1) 4h HMA(21) for macro trend bias - only take longs above, shorts below
2) 30m Vol Spike detection - ATR(7)/ATR(30) > 1.8 signals panic/extreme move
3) Bollinger Band extremes - BB(20, 2.5) for mean reversion entry
4) ADX(14) regime filter - ADX < 25 = range (mean revert), ADX > 25 = trend (follow)
5) Volume confirmation - volume > 1.3x 20-bar avg (filters false signals)
6) Session filter - only 6-22 UTC (high liquidity, lower slippage)
7) ATR(14) trailing stop at 2.0x - tight stop for mean reversion
8) Position size: 0.20 base, 0.30 max with full confluence

Why 30m works here:
- Vol spikes happen intraday, 30m captures them better than 4h/1d
- HTF (4h) filter prevents counter-trend trades
- Mean reversion has higher win rate than trend-following in 2022-2025 bear/range
- Target: 40-80 trades/year (strict confluence = low fee drag)

Position sizing: discrete levels (0.0, ±0.20, ±0.30) to minimize churn
Stoploss: 2.0*ATR trailing (tighter for mean reversion)
Take profit: exit at BB mid or RSI(14) > 65 (long) / < 35 (short)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_volspike_bb_meanrev_4h1d_v1"
timeframe = "30m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate DM and TR
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to hours UTC
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
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Vol spike ratio: ATR(7) / ATR(30)
    vol_spike_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            vol_spike_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_spike_ratio[i] = 0.0
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (6-22 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = (hour >= 6) and (hour <= 22)
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_spike_ratio[i] > 1.8  # Panic/extreme move
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_below_bb_mid = close[i] < bb_mid[i]
        price_above_bb_mid = close[i] > bb_mid[i]
        
        # === ADX REGIME ===
        adx_low = adx_14[i] < 25  # Range market - mean revert
        adx_high = adx_14[i] >= 25  # Trending market - follow trend
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        rsi_neutral_long = rsi_14[i] < 50
        rsi_neutral_short = rsi_14[i] > 50
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.3
        volume_strong = volume_ratio > 1.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY (Mean Reversion in Range OR Pullback in Trend) ---
        # Confluence: 4h trend up + vol spike + price below BB + RSI oversold + volume
        if price_above_hma_4h and in_session:
            if adx_low:  # Range market - pure mean reversion
                if vol_spike and price_below_bb_lower and rsi_oversold:
                    if volume_confirmed:
                        new_signal = POSITION_SIZE_BASE
                        if volume_strong and rsi_14[i] < 25:
                            new_signal = POSITION_SIZE_MAX
            elif adx_high:  # Trending market - pullback entry
                if price_below_bb_mid and rsi_neutral_long and rsi_oversold:
                    if volume_confirmed:
                        new_signal = POSITION_SIZE_BASE
                        if volume_strong:
                            new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY (Mean Reversion in Range OR Pullback in Trend) ---
        # Confluence: 4h trend down + vol spike + price above BB + RSI overbought + volume
        if price_below_hma_4h and in_session:
            if adx_low:  # Range market - pure mean reversion
                if vol_spike and price_above_bb_upper and rsi_overbought:
                    if volume_confirmed:
                        new_signal = -POSITION_SIZE_BASE
                        if volume_strong and rsi_14[i] > 75:
                            new_signal = -POSITION_SIZE_MAX
            elif adx_high:  # Trending market - pullback entry
                if price_above_bb_mid and rsi_neutral_short and rsi_overbought:
                    if volume_confirmed:
                        new_signal = -POSITION_SIZE_BASE
                        if volume_strong:
                            new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold long if still below BB mid and 4h trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if price_below_bb_mid and price_above_hma_4h and in_session:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if price_above_bb_mid and price_below_hma_4h and in_session:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT: Exit at BB Mid ===
        if in_position and position_side > 0:
            if price_above_bb_mid:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_below_bb_mid:
                new_signal = 0.0
        
        # === TAKE PROFIT: Exit on RSI Extreme ===
        if in_position and position_side > 0 and rsi_14[i] > 65:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals