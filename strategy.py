#!/usr/bin/env python3
"""
Experiment #344: 30m KAMA Adaptive Trend + 4h HMA Macro + RSI Pullback + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than 
fixed EMAs, performing well in both trending and ranging conditions. Combined with 4h HMA 
for macro trend bias (faster than daily, more reliable than 1h), and RSI pullback entries 
(proven in best strategy mtf_12h_supertrend_daily_hma_rsi_pullback_v2). This should generate 
more trades than 12h strategies while maintaining quality through HTF filter.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 40-80 trades/year, adaptive trend following with pullback entries.
Key insight: KAMA efficiency ratio filters noise, 4h HMA provides trend bias, RSI pullback = quality entries.
Position sizing: 0.25 entry, 0.125 half (take profit), stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_rsi_pullback_atr_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility using Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER = trending (fast SC), Low ER = ranging (slow SC)
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    net_change = np.abs(close - np.roll(close, period))
    net_change[:period] = np.nan
    
    volatility = np.zeros(len(close))
    for i in range(period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    er = np.zeros(len(close))
    er[period:] = np.where(volatility[period:] > 0, net_change[period:] / volatility[period:], 0)
    er[:period] = np.nan
    
    # Calculate Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er ** 2 * (fast_sc - slow_sc) + slow_sc
    sc[:period] = np.nan
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[period] = close[period]  # Initialize with price
    
    for i in range(period + 1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # Additional KAMA for trend confirmation (slower)
    kama_slow = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    
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
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # KAMA trend direction (fast vs slow)
        kama_bullish = kama[i] > kama_slow[i]
        kama_bearish = kama[i] < kama_slow[i]
        
        # KAMA slope (price vs KAMA)
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI pullback levels (entry on weakness in uptrend, strength in downtrend)
        rsi_pullback_long = rsi[i] < 45  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 55  # Pullback in downtrend
        rsi_extreme_long = rsi[i] < 35  # Deep pullback
        rsi_extreme_short = rsi[i] > 65  # Deep rally
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40  # Not too weak
        rsi_momentum_short = rsi[i] < 60  # Not too strong
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: 4h bullish + KAMA bullish + RSI pullback
        if trend_bullish and kama_bullish and rsi_pullback_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + Price above KAMA + RSI extreme (deep pullback entry)
        elif trend_bullish and price_above_kama and rsi_extreme_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA crossover bullish + RSI ok (momentum entry)
        elif kama_bullish and price_above_kama and rsi[i] > 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: 4h bearish + KAMA bearish + RSI pullback
        if trend_bearish and kama_bearish and rsi_pullback_short and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + Price below KAMA + RSI extreme (deep rally entry)
        elif trend_bearish and price_below_kama and rsi_extreme_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA crossover bearish + RSI ok (momentum entry)
        elif kama_bearish and price_below_kama and rsi[i] < 50:
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