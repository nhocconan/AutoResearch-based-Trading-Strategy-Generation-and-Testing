#!/usr/bin/env python3
"""
Experiment #342: 1d KAMA Adaptive Trend + Weekly HMA + Choppiness Regime + ATR Stop
Hypothesis: Daily timeframe with KAMA (Kaufman Adaptive Moving Average) adapts to
market volatility - fast during trends, slow during ranges. Combined with weekly
HMA for macro bias and Choppiness Index to avoid trading in choppy markets.
This should reduce whipsaws in 2022 crash while capturing trends in 2021 bull.
Timeframe: 1d (REQUIRED), HTF: 1w for macro trend via mtf_data helper.
Target: Beat Sharpe=0.499 with 15-30 trades/year, adaptive trend following.
Key insight: KAMA efficiency ratio filters noise, CHOP avoids range markets,
weekly HMA provides macro bias to avoid counter-trend trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_hma_chop_regime_atr_v1"
timeframe = "1d"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - fast in trends, slow in ranges.
    Efficiency Ratio (ER) = |net change| / sum of absolute changes
    Smoothing Constant = (ER * (fast SC - slow SC) + slow SC)^2
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    net_change = np.abs(close - np.roll(close, period))
    net_change[:period] = np.nan
    
    sum_changes = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    sum_changes[:period] = np.nan
    
    er = np.where(sum_changes > 0, net_change / sum_changes, 0.0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[period] = close[period]  # Initialize with first valid close
    
    for i in range(period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:period] = np.nan
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    # True range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high - lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Fast KAMA for crossover signals
    kama_fast = calculate_kama(close, period=5, fast_period=2, slow_period=20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(kama[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Choppiness regime filter
        trending_market = chop[i] < 45.0  # Below 45 = trending (looser than 38.2)
        ranging_market = chop[i] > 55.0  # Above 55 = ranging
        
        # KAMA trend direction
        kama_bullish = kama_fast[i] > kama[i] and kama[i] > kama[i-1] if not np.isnan(kama[i-1]) else False
        kama_bearish = kama_fast[i] < kama[i] and kama[i] < kama[i-1] if not np.isnan(kama[i-1]) else False
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1] if not np.isnan(kama_fast[i-1]) else False
        kama_cross_short = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1] if not np.isnan(kama_fast[i-1]) else False
        
        # Price vs KAMA position
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum filter
        rsi_ok_long = rsi[i] > 40 and rsi[i] < 75  # Not oversold, not overbought
        rsi_ok_short = rsi[i] < 60 and rsi[i] > 25  # Not overbought, not oversold
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: KAMA cross + Weekly bullish + Trending market
        if kama_cross_long and weekly_bullish and trending_market:
            new_signal = SIZE_ENTRY
        # Secondary: Price above KAMA + Weekly bullish + RSI ok
        elif price_above_kama and weekly_bullish and rsi_ok_long and trending_market:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA bullish + RSI momentum (no weekly filter for more trades)
        elif kama_bullish and rsi[i] > 50 and trending_market:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: KAMA cross + Weekly bearish + Trending market
        if kama_cross_short and weekly_bearish and trending_market:
            new_signal = -SIZE_ENTRY
        # Secondary: Price below KAMA + Weekly bearish + RSI ok
        elif price_below_kama and weekly_bearish and rsi_ok_short and trending_market:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA bearish + RSI momentum (no weekly filter for more trades)
        elif kama_bearish and rsi[i] < 50 and trending_market:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals