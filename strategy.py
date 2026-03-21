#!/usr/bin/env python3
"""
Experiment #013: 15m Multi-Timeframe Regime Adaptive Strategy
Hypothesis: 15m timeframe captures intraday swings while 4h/1h HTF provides trend context.
Use Choppiness Index to detect range vs trend regimes - trend follow in trending markets,
mean revert in ranging markets. 4h HMA gives major trend direction, 1h RSI for entry timing.
ATR-based stoploss (2.5x) protects against crashes. Position sizing 0.25-0.30 discrete.
This should work better in 2025 bear/range market than pure trend following.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_rsi_4h_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index - high values = ranging, low = trending."""
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    price_range = highest_high - lowest_low
    
    chop = np.zeros(len(close))
    mask = price_range > 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(er_period))
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA for major trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h RSI for entry timing
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    rsi_15m = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.mean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0 and not np.isnan(hma_4h_aligned[i])
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 15m HMA trend
        hma_trend_long = hma_fast[i] > hma_slow[i]
        hma_trend_short = hma_fast[i] < hma_slow[i]
        
        # Choppiness regime detection
        is_trending = chop[i] < 50  # Below 50 = trending
        is_ranging = chop[i] > 50   # Above 50 = ranging
        
        # 1h RSI for entry timing
        rsi_1h_valid = not np.isnan(rsi_1h_aligned[i])
        rsi_1h_pullback_long = rsi_1h_valid and rsi_1h_aligned[i] > 35 and rsi_1h_aligned[i] < 55
        rsi_1h_pullback_short = rsi_1h_valid and rsi_1h_aligned[i] > 45 and rsi_1h_aligned[i] < 65
        
        # 15m RSI for mean reversion entries
        rsi_15m_oversold = rsi_15m[i] < 35
        rsi_15m_overbought = rsi_15m[i] > 65
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # KAMA slope for trend confirmation
        kama_slope_long = kama[i] > kama[i-3] if i > 3 else True
        kama_slope_short = kama[i] < kama[i-3] if i > 3 else True
        
        # Entry logic - regime adaptive
        new_signal = 0.0
        
        # TRENDING REGIME: Follow 4h trend with 15m HMA confirmation
        if is_trending:
            # Long in bullish 4h trend
            if trend_4h_bullish and hma_trend_long and kama_slope_long:
                if rsi_1h_pullback_long or rsi_15m[i] < 50:
                    new_signal = SIZE
            # Short in bearish 4h trend
            elif trend_4h_bearish and hma_trend_short and kama_slope_short:
                if rsi_1h_pullback_short or rsi_15m[i] > 50:
                    new_signal = -SIZE
        
        # RANGING REGIME: Mean reversion with RSI extremes
        elif is_ranging:
            # Long on oversold RSI
            if rsi_15m_oversold and close[i] > kama[i]:
                new_signal = SIZE
            # Short on overbought RSI
            elif rsi_15m_overbought and close[i] < kama[i]:
                new_signal = -SIZE
        
        # HMA crossover entries (work in both regimes)
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        if hma_cross_long and vol_confirm and trend_4h_bullish:
            new_signal = SIZE
        elif hma_cross_short and vol_confirm and trend_4h_bearish:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                current_trail = close[i] - 2.5 * atr[i]
                if current_trail > trailing_stop[i-1] if i > 0 else 0:
                    trailing_stop[i] = current_trail
                else:
                    trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
                
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] > entry_price[i-1] + 2.5 * atr[i] and new_signal == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                current_trail = close[i] + 2.5 * atr[i]
                if current_trail < trailing_stop[i-1] if i > 0 else 999999:
                    trailing_stop[i] = current_trail
                else:
                    trailing_stop[i] = trailing_stop[i-1] if i > 0 else 999999
                
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] < entry_price[i-1] - 2.5 * atr[i] and new_signal == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            else:
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals