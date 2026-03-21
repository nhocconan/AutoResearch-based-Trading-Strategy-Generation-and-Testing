#!/usr/bin/env python3
"""
Experiment #038: 30m Choppiness Regime + 4h HMA Trend + RSI Mean Reversion
Hypothesis: 30m timeframe is too noisy for pure trend following (as seen in exp#032, #037).
Instead, use Choppiness Index to detect range vs trend regime, then apply appropriate logic:
- Range (CHOP > 61.8): Mean reversion with RSI extremes (buy <30, sell >70)
- Trend (CHOP < 38.2): Trend following with pullback entries (RSI 40-60 in direction)
4h HMA provides major trend filter to avoid counter-trend trades.
This regime-adaptive approach should work better on 30m than pure trend strategies.
Position sizing 0.25 with 2.5x ATR stoploss for crash protection.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_regime_4h_hma_rsi_v2"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = Range/Consolidation (mean reversion favorable)
    CHOP < 38.2 = Strong Trend (trend following favorable)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i-er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        er[i] = change / noise if noise > 0 else 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bull = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bear = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Choppiness regime detection
        chop_range = chop[i] > 55  # Range/consolidation (relaxed from 61.8)
        chop_trend = chop[i] < 45  # Strong trend (relaxed from 38.2)
        chop_neutral = not chop_range and not chop_trend
        
        # KAMA trend direction
        kama_rising = kama[i] > kama[i-5] if i > 5 else False
        kama_falling = kama[i] < kama[i-5] if i > 5 else False
        
        # HMA alignment
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # RSI signals
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral_long = 40 < rsi[i] < 60
        rsi_neutral_short = 40 < rsi[i] < 60
        
        # Volume confirmation
        vol_above = volume[i] > vol_sma[i] * 0.9 if vol_sma[i] > 0 else True
        
        # Price vs KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Entry logic - REGIME ADAPTIVE
        new_signal = 0.0
        
        # === RANGE REGIME (Mean Reversion) ===
        if chop_range:
            # Long: RSI oversold + 4h not strongly bearish + price near support
            if rsi_oversold and not trend_4h_bear and price_above_kama:
                new_signal = SIZE
            # Short: RSI overbought + 4h not strongly bullish + price near resistance
            elif rsi_overbought and not trend_4h_bull and price_below_kama:
                new_signal = -SIZE
        
        # === TREND REGIME (Trend Following) ===
        elif chop_trend:
            # Long: 4h bullish + KAMA rising + RSI pullback (not oversold)
            if trend_4h_bull and kama_rising and rsi_neutral_long and hma_bull:
                new_signal = SIZE
            # Long: 4h bullish + HMA aligned + volume confirmation
            elif trend_4h_bull and hma_bull and vol_above and price_above_kama:
                new_signal = SIZE
            # Short: 4h bearish + KAMA falling + RSI pullback (not overbought)
            elif trend_4h_bear and kama_falling and rsi_neutral_short and hma_bear:
                new_signal = -SIZE
            # Short: 4h bearish + HMA aligned + volume confirmation
            elif trend_4h_bear and hma_bear and vol_above and price_below_kama:
                new_signal = -SIZE
        
        # === NEUTRAL REGIME (Conservative) ===
        else:
            # Only enter on strong 4h alignment + RSI confirmation
            if trend_4h_bull and rsi[i] > 50 and hma_bull:
                new_signal = SIZE * 0.8  # Reduced size in neutral
            elif trend_4h_bear and rsi[i] < 50 and hma_bear:
                new_signal = -SIZE * 0.8
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                if close[i] > entry_price + 2.5 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                if close[i] < entry_price - 2.5 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals