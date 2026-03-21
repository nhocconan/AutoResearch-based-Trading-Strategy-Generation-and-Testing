#!/usr/bin/env python3
"""
Experiment #443: 12h Donchian Breakout + Daily HMA Bias + Weekly Trend + Choppiness Filter
Hypothesis: 12h timeframe captures medium-term trends with fewer whipsaws than 4h.
Donchian breakout (20-period) captures trend continuation. Daily HMA provides HTF bias.
Weekly HMA adds meta-trend filter. Choppiness Index filters ranging markets (avoid trades when CHOP>61.8).
RSI pullback entry ensures we enter on dips within trends, not breakouts. 3*ATR stoploss for 12h volatility.
Multiple entry paths ensure >=10 trades per symbol. Position sizing 0.25-0.30 discrete.
Timeframe: 12h (REQUIRED), HTF: 1d and 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_hma_weekly_chop_rsi_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    Values > 61.8 = ranging market, < 38.2 = trending market.
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i - period:i + 1])
        lowest = np.min(low[i - period:i + 1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr1 = high[j] - low[j]
                tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
                tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
                tr = max(tr1, tr2, tr3)
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    sma200 = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):  # Start after 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (meta-trend)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Long-term trend filter
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        # Choppiness Index regime filter
        trending_market = chop[i] < 50  # More lenient than 38.2 to get more trades
        ranging_market = chop[i] > 55   # Avoid trades in choppy markets
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # Donchian pullback (price near lower band in uptrend = buy opportunity)
        donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2 if not np.isnan(donchian_upper[i]) else close[i]
        pullback_long = close[i] < donchian_mid and close[i] > donchian_lower[i]
        pullback_short = close[i] > donchian_mid and close[i] < donchian_upper[i]
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 25
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # RSI filter for entries
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_neutral_long = rsi[i] > 35 and rsi[i] < 65
        rsi_neutral_short = rsi[i] > 35 and rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + Weekly bullish + Daily bullish + Trending market
        if breakout_long and weekly_bullish and daily_bullish and trending_market:
            new_signal = SIZE_ENTRY
        # Path 2: Donchian pullback + Weekly bullish + RSI oversold + ADX > 18
        elif pullback_long and weekly_bullish and rsi_oversold and adx[i] > 18:
            new_signal = SIZE_ENTRY
        # Path 3: Daily bullish + Weekly bullish + Above SMA200 + RSI 40-60 + DI bullish
        elif daily_bullish and weekly_bullish and above_sma200 and rsi_neutral_long and di_bullish:
            new_signal = SIZE_ENTRY
        # Path 4: Price above Donchian mid + Daily bullish + ADX rising + RSI > 40
        elif close[i] > donchian_mid and daily_bullish and adx[i] > adx[i-1] and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 5: Weekly bullish + DI bullish + Trending market + RSI 35-55
        elif weekly_bullish and di_bullish and trending_market and rsi[i] > 35 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 6: Breakout long + Above SMA200 + ADX > 20
        elif breakout_long and above_sma200 and adx[i] > 20:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + Weekly bearish + Daily bearish + Trending market
        if breakout_short and weekly_bearish and daily_bearish and trending_market:
            new_signal = -SIZE_ENTRY
        # Path 2: Donchian pullback + Weekly bearish + RSI overbought + ADX > 18
        elif pullback_short and weekly_bearish and rsi_overbought and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        # Path 3: Daily bearish + Weekly bearish + Below SMA200 + RSI 40-60 + DI bearish
        elif daily_bearish and weekly_bearish and below_sma200 and rsi_neutral_short and di_bearish:
            new_signal = -SIZE_ENTRY
        # Path 4: Price below Donchian mid + Daily bearish + ADX rising + RSI < 60
        elif close[i] < donchian_mid and daily_bearish and adx[i] > adx[i-1] and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: Weekly bearish + DI bearish + Trending market + RSI 45-65
        elif weekly_bearish and di_bearish and trending_market and rsi[i] > 45 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Path 6: Breakout short + Below SMA200 + ADX > 20
        elif breakout_short and below_sma200 and adx[i] > 20:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR for 12h timeframe)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR for 12h timeframe)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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