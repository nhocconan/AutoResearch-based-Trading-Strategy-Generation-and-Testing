#!/usr/bin/env python3
"""
Experiment #045: 1h Multi-Confluence with 4h/1d HTF Filters

Hypothesis: 1h entries with EXTREME selectivity (4+ confluence filters) will
generate few enough trades (30-60/year) to overcome fee drag while capturing
high-probability setups. Key insight from failures: lower TF needs HARDER filters.

Key design:
1. 4h HMA(21) for intermediate trend bias (call ONCE via mtf_data)
2. 1d ADX(14) for regime strength (>25 = trend, <20 = range)
3. 1h RSI(7) for fast entry timing (extreme only: <25 or >75)
4. Volume filter: >1.2x 20-bar avg (confirms conviction)
5. Session filter: 8-20 UTC only (liquid hours, avoid Asia whipsaw)
6. ATR(14) stoploss at 2.5x (tight for 1h)
7. Discrete sizing: 0.20 base (smaller for lower TF fee sensitivity)

Why this should work:
- 1h TF with 4+ filters = ~40-60 trades/year (fee efficient)
- 4h HTF prevents counter-trend trades
- 1d ADX regime adapts entry thresholds
- RSI(7) extreme only = high win rate entries
- Volume + session = avoids fake breakouts
- Small size (0.20) limits drawdown on 1h noise

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data helper
Position sizing: 0.20 discrete (smaller for 1h fee sensitivity)
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4confluence_4h_1d_hma_adx_rsi_vol_session_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    # Smooth with Wilder's method
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    di_sum = plus_di + minus_di
    di_sum = np.where(di_sum == 0, 1e-10, di_sum)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    
    # Calculate 4h HMA trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d ADX regime
    adx_1d_14 = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Fast RSI for 1h entries
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, smaller for 1h)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(adx_1d_aligned[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Convert open_time to hour (Binance uses ms timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1d ADX) ===
        # ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        # === VOLUME FILTER ===
        vol_confirmed = volume[i] > 1.2 * vol_avg[i]
        
        # === ENTRY LOGIC - EXTREME SELECTIVITY (4+ confluence) ===
        new_signal = 0.0
        
        if is_trending and htf_bullish:
            # Trend follow long: RSI(7) pullback + volume + session
            if rsi_7[i] < 30 and in_session and vol_confirmed:
                new_signal = STRONG_SIZE
        
        elif is_trending and htf_bearish:
            # Trend follow short: RSI(7) rally + volume + session
            if rsi_7[i] > 70 and in_session and vol_confirmed:
                new_signal = -STRONG_SIZE
        
        elif is_ranging:
            # Mean reversion in range: RSI(7) extreme + volume + session
            if rsi_7[i] < 25 and in_session and vol_confirmed:
                new_signal = BASE_SIZE  # long at extreme oversold
            elif rsi_7[i] > 75 and in_session and vol_confirmed:
                new_signal = -BASE_SIZE  # short at extreme overbought
        
        else:
            # Neutral regime: require VERY strong signal
            if htf_bullish and rsi_7[i] < 20 and in_session and vol_confirmed:
                new_signal = BASE_SIZE
            elif htf_bearish and rsi_7[i] > 80 and in_session and vol_confirmed:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~40 hours on 1h), allow easier entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if htf_bullish and rsi_7[i] < 35 and in_session:
                new_signal = BASE_SIZE * 0.8
            elif htf_bearish and rsi_7[i] > 65 and in_session:
                new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI(14) becomes very overbought
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            # Exit short when RSI(14) becomes very oversold
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals