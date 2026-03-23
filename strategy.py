#!/usr/bin/env python3
"""
Experiment #005: 1h Primary + 4h/1d HTF — Regime-Adaptive with Session/Volume Filter

Hypothesis: Previous failures came from either too tight confluence (0 trades) or 
no regime detection (whipsaw in 2022). This strategy uses Choppiness Index to 
detect regime, then applies DIFFERENT entry logic per regime:
- CHOP > 55 (range): Mean reversion at BB extremes + RSI extremes
- CHOP < 45 (trend): Pullback entries in HTF trend direction

Key improvements over failed attempts:
1. 1h timeframe with 4h/1d HTF trend filter (proven 2x Sharpe in research)
2. Session filter: Only trade 8-20 UTC (highest liquidity, lowest slippage)
3. Volume confirmation: volume > 0.8x 20-bar average
4. Loose enough entries to generate 40-80 trades/year (not 0, not 200+)
5. Position size 0.25 (conservative for 1h per Rule 4)
6. Stoploss: 2.5*ATR trailing

Entry conditions (designed for trade frequency):
- Long range: CHOP>55 + BB_pct_b<0.15 + RSI<30 + volume_ok + session_ok + 4h HMA not bearish
- Long trend: CHOP<45 + RSI 35-50 pullback + price>4h HMA + volume_ok + session_ok
- Short range: CHOP>55 + BB_pct_b>0.85 + RSI>70 + volume_ok + session_ok + 4h HMA not bullish
- Short trend: CHOP<45 + RSI 50-65 pullback + price<4h HMA + volume_ok + session_ok

Why this might beat Sharpe=0.366:
- Regime-adaptive (works in both 2021 bull and 2022 crash)
- Session filter reduces noise trades
- 1h entries within 4h trend = best of both worlds
- Loose enough to generate trades, strict enough to avoid churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_session_volume_4h1d_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # %B (position within bands)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    # Bandwidth for choppiness alternative
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    return upper.values, lower.values, pct_b.values, bandwidth.values

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
    
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    atr_vals = calculate_atr(high, low, close, period=period)
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    price_range = hh - ll + 1e-10
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for regime confirmation
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_pct_b, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume 20-bar average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract UTC hour for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(bb_pct_b[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(vol_avg[i]) or atr_14[i] == 0 or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = utc_hours[i]
        session_ok = (hour >= 8) and (hour <= 20)
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === 4H TREND BIAS ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D REGIME CONFIRMATION ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        price_below_hma_1d = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_range = chop_value > 55  # Range/choppy market
        is_trend = chop_value < 45  # Trending market
        # 45-55 is neutral/transition
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = bb_pct_b[i] < 0.15
        bb_extreme_high = bb_pct_b[i] > 0.85
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        rsi_pullback_long = 35 <= rsi_14[i] <= 50
        rsi_pullback_short = 50 <= rsi_14[i] <= 65
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME (Mean Reversion) ---
        if is_range and session_ok and volume_ok:
            # Long: BB low + RSI oversold + 4H not strongly bearish
            if bb_extreme_low and rsi_oversold:
                if not (hma_4h_slope_bear and price_below_hma_4h):
                    new_signal = POSITION_SIZE
            
            # Short: BB high + RSI overbought + 4H not strongly bullish
            if bb_extreme_high and rsi_overbought:
                if not (hma_4h_slope_bull and price_above_hma_4h):
                    new_signal = -POSITION_SIZE
        
        # --- TREND REGIME (Pullback Entries) ---
        if is_trend and session_ok and volume_ok:
            # Long pullback: RSI 35-50 + price above 4H HMA + 4H bullish
            if rsi_pullback_long and price_above_hma_4h and hma_4h_slope_bull:
                new_signal = POSITION_SIZE
            
            # Short pullback: RSI 50-65 + price below 4H HMA + 4H bearish
            if rsi_pullback_short and price_below_hma_4h and hma_4h_slope_bear:
                new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME (45-55 CHOP) ---
        # Only take high-conviction mean reversion
        if not is_range and not is_trend and session_ok and volume_ok:
            if bb_extreme_low and rsi_14[i] < 25:
                if price_above_hma_1d:  # Only long if above daily HMA
                    new_signal = POSITION_SIZE
            if bb_extreme_high and rsi_14[i] > 75:
                if price_below_hma_1d:  # Only short if below daily HMA
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals