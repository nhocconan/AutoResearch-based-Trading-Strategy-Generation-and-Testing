#!/usr/bin/env python3
"""
Experiment #090: 1h Primary + 4h/12h HTF — Regime-Adaptive Pullback Strategy

Hypothesis: 1h strategies failed (#085) due to overly strict confluence causing 0 trades.
This version uses PROVEN pattern: HTF trend direction + LTF pullback entries + regime filter.

Key design:
1) 12h HMA = macro trend bias (long only if price > 12h HMA)
2) 4h HMA = intermediate trend confirmation
3) 1h RSI = entry timing (pullback to 35-45 in uptrend, rally to 55-65 in downtrend)
4) Choppiness Index = regime filter (CHOP>50 = range → mean revert, CHOP<50 = trend → follow)
5) Session filter = only 8-20 UTC (high volume hours, avoid Asia low-vol)
6) Volume confirmation = volume > 0.8x 20-period avg
7) ATR(14) trailing stoploss at 2.5x

Why this should work on 1h:
- HTF filters reduce trade frequency to 30-80/year (not 200+)
- Pullback entries have better R:R than breakout entries
- Session filter avoids whipsaw during low-volume hours
- Regime-adaptive: different logic for range vs trend markets
- Discrete sizing (0.20/0.30) minimizes fee churn

Position size: 0.20 base, 0.30 max with confluence
Stoploss: 2.5*ATR trailing
Target: 40-70 trades/year, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h12h_chop_session_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling volume average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
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
    
    # Calculate 12h HMA for macro trend
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h HMA for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_1h_16 = calculate_hma(close, period=16)
    hma_1h_48 = calculate_hma(close, period=48)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Extract UTC hours for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_16[i]) or np.isnan(hma_1h_48[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(ema_21[i]) or np.isnan(vol_avg_20[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === HTF TREND BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0  # trending market
        chop_ranging = chop_14[i] >= 50.0  # ranging market
        
        # === 1h HMA CROSSOVER ===
        hma_bullish = hma_1h_16[i] > hma_1h_48[i]
        hma_bearish = hma_1h_16[i] < hma_1h_48[i]
        
        # === RSI ENTRY SIGNALS (Pullback entries) ===
        # In uptrend: buy pullback when RSI 35-45
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        # In downtrend: sell rally when RSI 50-65
        rsi_rally_short = 50.0 <= rsi_14[i] <= 65.0
        # Extreme mean reversion (range market)
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = close[i] > ema_21[i]
        ema_bearish = close[i] < ema_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: HTF uptrend + 1h pullback + session + volume ---
        # Primary: 12h bullish + 4h bullish + RSI pullback + session + volume
        if price_above_hma_12h and price_above_hma_4h and in_session and volume_ok:
            if chop_trending:
                # Trending regime: follow trend on pullback
                if rsi_pullback_long and hma_bullish and ema_bullish:
                    new_signal = POSITION_SIZE_BASE
                    # Boost with extra confluence
                    if rsi_14[i] < 40.0:
                        new_signal = POSITION_SIZE_MAX
            elif chop_ranging:
                # Ranging regime: mean revert at oversold
                if rsi_oversold or (rsi_pullback_long and close[i] < ema_21[i] * 0.98):
                    new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY: HTF downtrend + 1h rally + session + volume ---
        # Primary: 12h bearish + 4h bearish + RSI rally + session + volume
        if price_below_hma_12h and price_below_hma_4h and in_session and volume_ok:
            if chop_trending:
                # Trending regime: follow trend on rally
                if rsi_rally_short and hma_bearish and ema_bearish:
                    new_signal = -POSITION_SIZE_BASE
                    # Boost with extra confluence
                    if rsi_14[i] > 60.0:
                        new_signal = -POSITION_SIZE_MAX
            elif chop_ranging:
                # Ranging regime: mean revert at overbought
                if rsi_overbought or (rsi_rally_short and close[i] > ema_21[i] * 1.02):
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if RSI hasn't reached extreme exit zone
        if in_position and new_signal == 0.0:
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
        # Exit long if 12h HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_12h:
                new_signal = 0.0
        
        # Exit short if 12h HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_12h:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
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