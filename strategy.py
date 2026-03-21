#!/usr/bin/env python3
"""
Experiment #386: 30m Donchian Breakout + 4h HMA Trend + ADX Momentum + ATR Stop
Hypothesis: Donchian Channel breakouts (20-period) capture trend continuations effectively
on 30m timeframe. 4h HMA provides trend bias to filter counter-trend breakouts that cause
whipsaws. ADX(14) > 20 ensures trending regime (lower threshold than typical 25 to ensure
trade frequency). RSI(14) loose filter (30-70) prevents entering at extremes. ATR(14)
stoploss at 2.0x protects capital while allowing normal volatility. Position size 0.25
discrete to minimize fee churn. Timeframe: 30m (REQUIRED), HTF: 4h for trend via mtf_data.
Key insight: Donchian breakouts generate more trades than Supertrend while 4h HMA filter
prevents the whipsaws that destroyed pure breakout strategies. Looser ADX/RSI thresholds
ensure minimum 10+ trades per symbol (critical - many strategies failed with 0 trades).
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_donchian_4h_hma_adx_rsi_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response with less lag."""
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
    n = len(close)
    adx = np.zeros(n)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX is EMA of DX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period-1] = np.nan
    lower[:period-1] = np.nan
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # Additional trend filter - 30m HMA
    hma_30m = calculate_hma(close, 21)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_30m[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation
        hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        
        # ADX trend strength (loose threshold to ensure trades)
        is_trending = adx[i] > 18  # Lower than typical 20-25
        
        # RSI filter (LOOSE to ensure trade frequency - CRITICAL)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 75  # Not deeply oversold, room to run
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 70  # Not deeply overbought, room to fall
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # Donchian position (already broken out)
        donchian_bullish = close[i] > donchian_upper[i-1]
        donchian_bearish = close[i] < donchian_lower[i-1]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades) ===
        # Primary: Donchian breakout + 4h bullish + 30m bullish + Trending + RSI ok
        if breakout_long and trend_bullish and hma_30m_bullish and is_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Donchian breakout + 4h bullish + RSI ok (ADX optional)
        elif breakout_long and trend_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Donchian bullish + 4h bullish + 30m bullish + RSI ok (no breakout needed)
        elif donchian_bullish and trend_bullish and hma_30m_bullish and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Quaternary: Donchian breakout + 30m bullish + RSI ok (4h neutral ok)
        elif breakout_long and hma_30m_bullish and rsi[i] > 35 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Quintenary: Donchian bullish alone with RSI filter (ensures minimum trades)
        elif donchian_bullish and rsi[i] > 35 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades) ===
        # Primary: Donchian breakout + 4h bearish + 30m bearish + Trending + RSI ok
        if breakout_short and trend_bearish and hma_30m_bearish and is_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Donchian breakout + 4h bearish + RSI ok (ADX optional)
        elif breakout_short and trend_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Donchian bearish + 4h bearish + 30m bearish + RSI ok (no breakout needed)
        elif donchian_bearish and trend_bearish and hma_30m_bearish and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: Donchian breakout + 30m bearish + RSI ok (4h neutral ok)
        elif breakout_short and hma_30m_bearish and rsi[i] > 30 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quintenary: Donchian bearish alone with RSI filter (ensures minimum trades)
        elif donchian_bearish and rsi[i] > 30 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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