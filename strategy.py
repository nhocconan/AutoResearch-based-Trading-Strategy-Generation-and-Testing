#!/usr/bin/env python3
"""
Experiment #018: 30m Volatility Breakout with HTF Trend Filter

Hypothesis: Previous strategies failed due to overly strict entry conditions
(Connors RSI extremes + Choppiness + Session filters = 0 trades).

This strategy uses VOLATILITY EXPANSION breakout instead of mean reversion:
1. 4h HMA(21) for trend direction (trade WITH HTF trend only)
2. 1d HMA(21) for major bias filter
3. Bollinger Band squeeze detection (BW percentile < 20)
4. Donchian(20) breakout for entry trigger
5. RSI(14) momentum confirmation (>55 for long, <45 for short)
6. ATR(14) trailing stoploss at 2.0x

Why this should work:
- Volatility squeeze → expansion is a proven breakout pattern
- HTF trend filter prevents counter-trend trades
- RSI momentum confirms genuine breakouts (not fakeouts)
- Fewer filters = more trades (target 40-80/year)
- Works in both bull and bear regimes

Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (conservative for 30m)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_squeeze_breakout_4h1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bw_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of bandwidth over lookback period."""
    bw_s = pd.Series(bandwidth)
    bw_pct = bw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= lookback else 0.5
    )
    return bw_pct.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    bw_percentile = calculate_bw_percentile(bb_bandwidth, 100)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_bandwidth[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(bw_percentile[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === VOLATILITY SQUEEZE DETECTION ===
        # BW percentile < 0.25 = volatility compression (setup for breakout)
        is_squeeze = bw_percentile[i] < 0.25
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above Donchian upper = bullish breakout
        # Price breaks below Donchian lower = bearish breakout
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI MOMENTUM CONFIRMATION ===
        # RSI > 55 = bullish momentum
        # RSI < 45 = bearish momentum
        rsi_bullish = rsi_14[i] > 55
        rsi_bearish = rsi_14[i] < 45
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma[i] if not np.isnan(volume_sma[i]) else True
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bullish + squeeze + breakout + RSI confirmation
        # Loosened: allow entry if 1d bullish OR (4h bullish + squeeze)
        if trend_4h_bullish or trend_1d_bullish:
            if (is_squeeze or bars_since_last_trade > 150) and breakout_long and rsi_bullish and volume_ok:
                new_signal = BASE_SIZE
        
        # SHORT ENTRY: 4h bearish + squeeze + breakout + RSI confirmation
        # Loosened: allow entry if 1d bearish OR (4h bearish + squeeze)
        if trend_4h_bearish or trend_1d_bearish:
            if (is_squeeze or bars_since_last_trade > 150) and breakout_short and rsi_bearish and volume_ok:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~3 days on 30m), allow weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and trend_1d_bullish and rsi_14[i] > 50 and close[i] > donchian_upper[i-5] if i > 5 else False:
                new_signal = BASE_SIZE * 0.6
            elif trend_4h_bearish and trend_1d_bearish and rsi_14[i] < 50 and close[i] < donchian_lower[i-5] if i > 5 else False:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bearish and rsi_14[i] < 40:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish and rsi_14[i] > 60:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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