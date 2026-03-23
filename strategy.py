#!/usr/bin/env python3
"""
Experiment #005: 1h Primary + 4h/1d HTF — Vol Spike Mean Reversion with Regime Filter

Hypothesis: 1h timeframe with strict confluence filters (HTF trend + vol spike + BB extremes + session)
will generate 30-80 trades/year while capturing volatility crush after panic spikes.

Key insight from failures: BTC/ETH fail on pure trend following (2022 crash, 2025 bear).
Mean reversion AFTER vol spikes works better in bear/range markets.

Strategy components:
1. 4h HMA(21): Macro trend bias (only trade WITH 4h trend for higher win rate)
2. 1d ADX(14): Regime filter (ADX>25 = trend, ADX<20 = range)
3. ATR(7)/ATR(30) ratio: Vol spike detection (>2.0 = panic, enter on reversion)
4. Bollinger Bands(20, 2.5): Entry at extremes (wider bands for fewer false signals)
5. RSI(14): Momentum confirmation (oversold/overbought extremes)
6. Session filter: Only 8-20 UTC (high volume hours, avoid Asian chop)
7. Volume filter: Volume > 0.8x 20-period avg (confirm participation)
8. ATR trailing stop: 2.5*ATR exit when vol normalizes

Why this should work on 1h:
- 4h/1d HTF filters = trade direction from higher TF (fewer whipsaws)
- Vol spike + BB extreme = high-probability mean reversion setup
- Session filter = avoids low-liquidity hours (reduces fake breakouts)
- Discrete position sizing (0.25) = limits drawdown, reduces fee churn

Position size: 0.25 (conservative for 1h TF)
Stoploss: 2.5*ATR trailing
Target trades: 30-80/year (strict confluence = 3+ filters must agree)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_volspike_bb_reversion_4h1d_v1"
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
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values (Wilder's smoothing)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with wider bands for fewer signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_bb_width(upper, lower, mid):
    """Calculate Bollinger Band Width (volatility measure)."""
    return (upper - lower) / (mid + 1e-10)

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
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d ADX for regime filter
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_mid)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Vol spike ratio
    vol_spike_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(vol_avg_20[i]) or atr_14[i] == 0:
            continue
        
        # Extract hour from open_time for session filter
        # open_time is in milliseconds, convert to hour
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hour_utc <= 20
        
        # === 4H TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D REGIME (ADX) ===
        adx_value = adx_1d_aligned[i]
        is_trending = adx_value > 25.0
        is_ranging = adx_value < 20.0
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_spike_ratio[i] > 2.0  # ATR(7) > 2x ATR(30)
        vol_normal = vol_spike_ratio[i] < 1.2  # Vol normalized
        
        # === BOLLINGER BAND EXTREMES ===
        price_at_bb_lower = close[i] <= bb_lower[i] * 1.005  # At or below lower band
        price_at_bb_upper = close[i] >= bb_upper[i] * 0.995  # At or above upper band
        price_near_mid = (bb_lower[i] * 1.02 < close[i] < bb_upper[i] * 0.98)  # Near middle
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30.0
        rsi_overbought = rsi_14[i] > 70.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > vol_avg_20[i] * 0.8
        
        # === ENTRY SIGNAL LOGIC (3+ confluence required) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Vol spike + BB lower + RSI oversold + 4h bullish + session ---
        if in_session and volume_ok:
            long_confluence = 0
            
            if price_at_bb_lower:
                long_confluence += 1
            if rsi_oversold:
                long_confluence += 1
            if vol_spike:
                long_confluence += 1
            if price_above_hma_4h:  # 4h trend supports long
                long_confluence += 1
            if rsi_rising:
                long_confluence += 1
            
            # Need 3+ confluence for long entry
            if long_confluence >= 3:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Vol spike + BB upper + RSI overbought + 4h bearish + session ---
        if in_session and volume_ok and new_signal == 0.0:
            short_confluence = 0
            
            if price_at_bb_upper:
                short_confluence += 1
            if rsi_overbought:
                short_confluence += 1
            if vol_spike:
                short_confluence += 1
            if price_below_hma_4h:  # 4h trend supports short
                short_confluence += 1
            if rsi_falling:
                short_confluence += 1
            
            # Need 3+ confluence for short entry
            if short_confluence >= 3:
                new_signal = -POSITION_SIZE
        
        # --- RANGING REGIME: Simpler mean reversion at BB extremes ---
        if is_ranging and new_signal == 0.0 and in_session and volume_ok:
            if price_at_bb_lower and rsi_oversold:
                new_signal = POSITION_SIZE
            elif price_at_bb_upper and rsi_overbought:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON VOL NORMALIZATION (take profit) ===
        if in_position and vol_normal and price_near_mid:
            # Vol normalized and price back to middle = take profit
            new_signal = 0.0
        
        # === EXIT ON 4H TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h and adx_value > 25:  # Strong bearish 4h trend
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and adx_value > 25:  # Strong bullish 4h trend
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
                # Flip position
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