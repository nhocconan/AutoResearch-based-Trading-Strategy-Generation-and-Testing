#!/usr/bin/env python3
"""
Experiment #301: 15m Multi-Timeframe Trend Following with Generous Entries
Hypothesis: 15m primary with 4h/1h trend filters captures intraday moves while avoiding whipsaw.
Key insight from failures: Entry conditions MUST be generous to ensure >=10 trades (learned from #289, #295, #296).
Using 4h HMA for macro bias + 1h RSI for timing + ADX to filter choppy markets.
Position size 0.25 (conservative), ATR stoploss at 2.5x, discrete levels to minimize fee churn.
Target: Beat Sharpe=0.499 while ensuring >=10 trades per symbol on 15m timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_adx_trend_atr_v1"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h indicators (macro trend)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    
    # Calculate 1h indicators (entry timing)
    hma_1h_21 = calculate_hma(df_1h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(
        df_1h['high'].values, 
        df_1h['low'].values, 
        df_1h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    hma_1h_21_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_21)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    plus_di_1h_aligned = align_htf_to_ltf(prices, df_1h, plus_di_1h)
    minus_di_1h_aligned = align_htf_to_ltf(prices, df_1h, minus_di_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    hma_15m_21 = calculate_hma(close, 21)
    hma_15m_50 = calculate_hma(close, 50)
    rsi_15m = calculate_rsi(close, 14)
    adx_15m, plus_di_15m, minus_di_15m = calculate_adx(high, low, close, 14)
    sma_15m_50 = calculate_sma(close, 50)
    
    # Track previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_hma_15m_21 = np.roll(hma_15m_21, 1)
    prev_hma_15m_21[0] = hma_15m_21[0]
    prev_rsi_15m = np.roll(rsi_15m, 1)
    prev_rsi_15m[0] = rsi_15m[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_15m_21[i]) or np.isnan(hma_15m_50[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(rsi_4h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_1h_aligned[i]) or np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 4H MACRO TREND BIAS ===
        # Bullish: Price > HMA21 > HMA50 on 4h
        trend_4h_bullish = (close[i] > hma_4h_21_aligned[i] and 
                           hma_4h_21_aligned[i] > hma_4h_50_aligned[i])
        # Bearish: Price < HMA21 < HMA50 on 4h
        trend_4h_bearish = (close[i] < hma_4h_21_aligned[i] and 
                           hma_4h_21_aligned[i] < hma_4h_50_aligned[i])
        
        # 4h RSI confirmation (not extreme)
        rsi_4h_ok_long = 35 < rsi_4h_aligned[i] < 75
        rsi_4h_ok_short = 25 < rsi_4h_aligned[i] < 65
        
        # === 1H ENTRY TIMING ===
        # 1h trend alignment
        trend_1h_bullish = close[i] > hma_1h_21_aligned[i]
        trend_1h_bearish = close[i] < hma_1h_21_aligned[i]
        
        # 1h RSI pullback (generous ranges for more trades)
        rsi_1h_pullback_long = 30 < rsi_1h_aligned[i] < 60
        rsi_1h_pullback_short = 40 < rsi_1h_aligned[i] < 70
        
        # 1h ADX filter (trend strength > 20 to avoid chop)
        adx_1h_strong = adx_1h_aligned[i] > 18
        
        # 1h DI crossover
        di_bullish = plus_di_1h_aligned[i] > minus_di_1h_aligned[i]
        di_bearish = plus_di_1h_aligned[i] < minus_di_1h_aligned[i]
        
        # === 15M LOCAL SIGNALS ===
        # 15m trend
        trend_15m_bullish = close[i] > hma_15m_21[i] and hma_15m_21[i] > hma_15m_50[i]
        trend_15m_bearish = close[i] < hma_15m_21[i] and hma_15m_21[i] < hma_15m_50[i]
        
        # 15m RSI (not extreme)
        rsi_15m_ok_long = 25 < rsi_15m[i] < 75
        rsi_15m_ok_short = 25 < rsi_15m[i] < 75
        
        # 15m HMA crossover
        hma_cross_long = prev_close[i] <= prev_hma_15m_21[i] and close[i] > hma_15m_21[i]
        hma_cross_short = prev_close[i] >= prev_hma_15m_21[i] and close[i] < hma_15m_21[i]
        
        # 15m ADX
        adx_15m_strong = adx_15m[i] > 15
        
        new_signal = 0.0
        
        # === LONG ENTRY (generous conditions for trades) ===
        # Primary: 4h bullish + 1h bullish + 1h RSI pullback + 15m cross
        if trend_4h_bullish and rsi_4h_ok_long and trend_1h_bullish and rsi_1h_pullback_long and hma_cross_long:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + 1h ADX strong + 15m trend + RSI ok
        elif trend_4h_bullish and rsi_4h_ok_long and adx_1h_strong and trend_15m_bullish and rsi_15m_ok_long:
            new_signal = SIZE_ENTRY
        # Tertiary: 4h bullish + 1h DI bullish + 15m cross (simpler for more trades)
        elif trend_4h_bullish and rsi_4h_ok_long and di_bullish and hma_cross_long:
            new_signal = SIZE_ENTRY
        # Quaternary: 4h bullish + 15m above SMA50 + RSI 40-65
        elif trend_4h_bullish and close[i] > sma_15m_50[i] and 40 < rsi_15m[i] < 65:
            new_signal = SIZE_ENTRY
        # Simple: 4h bullish + 1h bullish + 15m price > HMA21
        elif trend_4h_bullish and trend_1h_bullish and close[i] > hma_15m_21[i] and rsi_15m_ok_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY (generous conditions for trades) ===
        # Primary: 4h bearish + 1h bearish + 1h RSI pullback + 15m cross
        if trend_4h_bearish and rsi_4h_ok_short and trend_1h_bearish and rsi_1h_pullback_short and hma_cross_short:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + 1h ADX strong + 15m trend + RSI ok
        elif trend_4h_bearish and rsi_4h_ok_short and adx_1h_strong and trend_15m_bearish and rsi_15m_ok_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: 4h bearish + 1h DI bearish + 15m cross (simpler for more trades)
        elif trend_4h_bearish and rsi_4h_ok_short and di_bearish and hma_cross_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: 4h bearish + 15m below SMA50 + RSI 35-60
        elif trend_4h_bearish and close[i] < sma_15m_50[i] and 35 < rsi_15m[i] < 60:
            new_signal = -SIZE_ENTRY
        # Simple: 4h bearish + 1h bearish + 15m price < HMA21
        elif trend_4h_bearish and trend_1h_bearish and close[i] < hma_15m_21[i] and rsi_15m_ok_short:
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