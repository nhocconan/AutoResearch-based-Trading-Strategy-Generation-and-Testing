#!/usr/bin/env python3
"""
Experiment #176: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + ATR Stop

Hypothesis: Previous 12h strategies failed due to overly complex regime detection
and too many confluence filters (0 trades on BTC/ETH). Donchian breakouts are
proven to catch major moves in crypto (20%+ rallies, 50%+ crashes). Combined with
simple HMA trend filter and ATR trailing stops, this should generate consistent
trades across ALL symbols while maintaining positive Sharpe.

KEY IMPROVEMENTS over #172:
1. Donchian(20) breakout as PRIMARY signal - catches major trend moves
2. Removed Fisher Transform (inconsistent on BTC/ETH)
3. Removed complex regime detection (was blocking all entries)
4. Simpler HTF bias: 1d HMA only (not 1w)
5. Looser entry thresholds to ensure ≥30 trades per symbol on train
6. Better position holding: hold through pullbacks if trend intact
7. ATR trailing stop at 3.0x for risk management

TARGET: 25-40 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_atr_1d_v1"
timeframe = "12h"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = np.maximum(high_s - high_s.shift(1), 0).values
    minus_dm = np.maximum(low_s.shift(1) - low, 0).values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = np.nan_to_num(adx, nan=0.0)
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        current_price = close[i]
        prev_price = close[i-1] if i > 0 else current_price
        
        # === TREND BIAS FROM HTF ===
        price_above_hma_1d = current_price > hma_1d_aligned[i]
        price_below_hma_1d = current_price < hma_1d_aligned[i]
        
        # === 12H TREND ===
        price_above_hma_21 = current_price > hma_21[i]
        price_below_hma_21 = current_price < hma_21[i]
        hma_21_above_50 = hma_21[i] > hma_50[i] if not np.isnan(hma_50[i]) else False
        hma_21_below_50 = hma_21[i] < hma_50[i] if not np.isnan(hma_50[i]) else False
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper channel
        donchian_breakout_long = (current_price > donchian_upper[i]) and (prev_price <= donchian_upper[i-1] if i > 0 else False)
        # Breakout below lower channel
        donchian_breakout_short = (current_price < donchian_lower[i]) and (prev_price >= donchian_lower[i-1] if i > 0 else False)
        
        # Near breakout (within 1% of channel)
        near_breakout_long = current_price > donchian_upper[i] * 0.99
        near_breakout_short = current_price < donchian_lower[i] * 1.01
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 45.0
        rsi_bearish = rsi_14[i] < 55.0
        rsi_strong_bull = rsi_14[i] > 55.0
        rsi_strong_bear = rsi_14[i] < 45.0
        
        # === ADX TREND STRENGTH ===
        adx_trending = adx_14[i] > 18.0  # Lower threshold for more trades
        adx_strong = adx_14[i] > 25.0
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + trend confirmation
        if donchian_breakout_long or near_breakout_long:
            long_score = 0
            if price_above_hma_1d:
                long_score += 2
            if price_above_hma_21:
                long_score += 1
            if rsi_bullish:
                long_score += 1
            if adx_trending:
                long_score += 1
            if volume_ok:
                long_score += 1
            
            # Need score >= 3 for entry (loose enough for trades)
            if long_score >= 3:
                if adx_strong:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: Donchian breakdown + trend confirmation
        if donchian_breakout_short or near_breakout_short:
            short_score = 0
            if price_below_hma_1d:
                short_score += 2
            if price_below_hma_21:
                short_score += 1
            if rsi_bearish:
                short_score += 1
            if adx_trending:
                short_score += 1
            if volume_ok:
                short_score += 1
            
            # Need score >= 3 for entry
            if short_score >= 3:
                if adx_strong:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold long if trend still intact (price above HMA21)
        if in_position and position_side > 0 and new_signal == 0.0:
            if price_above_hma_21 and rsi_bullish:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # Hold short if trend still intact (price below HMA21)
        if in_position and position_side < 0 and new_signal == 0.0:
            if price_below_hma_21 and rsi_bearish:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, current_price)
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if current_price < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = current_price
            else:
                lowest_since_entry = min(lowest_since_entry, current_price)
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if current_price > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 12h HMA21 significantly
        if in_position and position_side > 0 and price_below_hma_21:
            # Only exit if RSI also turns bearish (avoid whipsaw)
            if rsi_14[i] < 45.0:
                new_signal = 0.0
        
        # Exit short if price crosses above 12h HMA21 significantly
        if in_position and position_side < 0 and price_above_hma_21:
            # Only exit if RSI also turns bullish
            if rsi_14[i] > 55.0:
                new_signal = 0.0
        
        # Exit if HTF bias flips strongly against position
        if in_position and position_side > 0 and price_below_hma_1d:
            if rsi_14[i] < 40.0:
                new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            if rsi_14[i] > 60.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = current_price
                entry_bar = i
                highest_since_entry = current_price if position_side > 0 else 0.0
                lowest_since_entry = current_price if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = current_price
                entry_bar = i
                highest_since_entry = current_price if position_side > 0 else 0.0
                lowest_since_entry = current_price if position_side < 0 else 0.0
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