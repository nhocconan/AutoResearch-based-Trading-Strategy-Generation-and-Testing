#!/usr/bin/env python3
"""
Experiment #105: 1h Primary + 4h/1d HTF — Fisher Transform + Volume + Session Filter

Hypothesis: Previous 1h strategies failed due to either too many trades (fee drag) or
insufficient confluence filters. This strategy combines:

1) 4h HMA(21) for macro trend direction (proven in #079, #101)
2) 1h Ehlers Fisher Transform(9) for precise reversal entries (better than RSI in bear)
3) 1d ADX(14) regime filter - only trade when ADX>20 (trending) or ADX<25 (range mean-revert)
4) Volume confirmation - entry only when volume > 1.3x 20-bar average
5) Session filter - only 8-20 UTC (highest liquidity, lowest slippage)
6) ATR-based dynamic position sizing - smaller size in high vol regimes

Why this should work:
- Fisher Transform catches reversals at extremes better than RSI (proven in literature)
- 4h HTF prevents counter-trend trades in 2022 crash and 2025 bear
- Volume + session filters reduce trade frequency to 40-70/year target
- ADX regime detection switches logic based on market state
- Conservative sizing (0.20-0.30) protects against 77% BTC crash

Position size: 0.20 base, 0.30 max with full confluence
Stoploss: 2.5*ATR trailing
Target: 40-70 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_vol_session_4h1d_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for better reversal detection.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    # Calculate typical price
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    # Normalize price to -1 to +1 range
    highest = typical_s.rolling(window=period, min_periods=period).max().values
    lowest = typical_s.rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    normalized = 2.0 * ((typical - lowest) / price_range) - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)  # Prevent log(0)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    # Smooth with Wilder's method
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100.0 * (plus_dm_smooth / (tr_smooth + 1e-10))
    minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return sma.values, upper.values, lower.values

def get_hour_from_open_time(open_time):
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 4h HMA slope
    hma_4h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-1]) and hma_4h_aligned[i-1] != 0:
            hma_4h_slope[i] = (hma_4h_aligned[i] - hma_4h_aligned[i-1]) / hma_4h_aligned[i-1] * 100
        else:
            hma_4h_slope[i] = 0.0
    
    # Calculate 1d ADX for regime detection
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Volume MA for confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract UTC hour for session filter
    utc_hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_bar = 0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(rsi_14[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        if np.isnan(bb_mid[i]) or np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (utc_hours[i] >= 8) and (utc_hours[i] <= 20)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.3 * volume_ma20[i]
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        hma_slope_positive = hma_4h_slope[i] > 0.3
        hma_slope_negative = hma_4h_slope[i] < -0.3
        
        # === 1D ADX REGIME ===
        adx_trending = adx_1d_aligned[i] > 25.0  # trending market
        adx_ranging = adx_1d_aligned[i] <= 25.0  # ranging market
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]  # crossing up from oversold
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]  # crossing down from overbought
        
        # Fisher crossover signals
        fisher_cross_up = (fisher[i] > fisher_signal[i]) and (fisher_signal[i] <= -1.0)
        fisher_cross_down = (fisher[i] < fisher_signal[i]) and (fisher_signal[i] >= 1.0)
        
        # === BOLLINGER BAND POSITION ===
        bb_range = bb_upper[i] - bb_lower[i] + 1e-10
        bb_pct = (close[i] - bb_lower[i]) / bb_range
        near_bb_lower = bb_pct < 0.15
        near_bb_upper = bb_pct > 0.85
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Must be in session and have volume confirmation
        if in_session and volume_confirmed:
            if adx_trending:
                # TREND MODE: Follow 4h trend with Fisher reversal
                if price_above_hma_4h and (hma_slope_positive or hma_4h_aligned[i] > hma_4h_aligned[i-1] if i > 0 else False):
                    if fisher_cross_up or (fisher_oversold and rsi_oversold):
                        new_signal = POSITION_SIZE_BASE
                        if fisher_cross_up and near_bb_lower:
                            new_signal = POSITION_SIZE_MAX
            else:
                # RANGE MODE: Mean revert at BB extremes with Fisher confirmation
                if near_bb_lower and (fisher_oversold or fisher_cross_up):
                    new_signal = POSITION_SIZE_BASE
                    if rsi_oversold and fisher_cross_up:
                        new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        if in_session and volume_confirmed:
            if adx_trending:
                # TREND MODE: Follow 4h trend with Fisher reversal
                if price_below_hma_4h and (hma_slope_negative or hma_4h_aligned[i] < hma_4h_aligned[i-1] if i > 0 else False):
                    if fisher_cross_down or (fisher_overbought and rsi_overbought):
                        new_signal = -POSITION_SIZE_BASE
                        if fisher_cross_down and near_bb_upper:
                            new_signal = -POSITION_SIZE_MAX
            else:
                # RANGE MODE: Mean revert at BB extremes with Fisher confirmation
                if near_bb_upper and (fisher_overbought or fisher_cross_down):
                    new_signal = -POSITION_SIZE_BASE
                    if rsi_overbought and fisher_cross_down:
                        new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if RSI not at extreme opposite
            if position_side > 0 and rsi_14[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_slope_negative:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            new_signal = 0.0
        
        # === EXIT AFTER 48 BARS (2 days) IF NO PROFIT ===
        if in_position and (i - entry_bar) > 48:
            if position_side > 0 and close[i] < entry_price * 1.01:
                new_signal = 0.0
            elif position_side < 0 and close[i] > entry_price * 0.99:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals