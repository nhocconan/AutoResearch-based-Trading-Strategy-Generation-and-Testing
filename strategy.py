#!/usr/bin/env python3
"""
Hypothesis: Daily (1d) primary with Weekly (1w) HTF trend filter captures major moves
while avoiding noise from lower timeframes. Supertrend(10,3) for entries + RSI(14)
pullback filter + volume confirmation. ATR(14) stoploss at 2.5*ATR protects capital.
SIZE=0.30 discrete levels balance trade frequency vs fee churn on daily bars.
Daily timeframe naturally produces fewer trades, so entry conditions are loosened
to ensure >=10 trades/symbol on train and >=3 on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_vol_1d_v1"
timeframe = "1d"
leverage = 1.0

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator - trend following with ATR-based stops"""
    n = len(close)
    
    # ATR calculation
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Supertrend values
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if close[i] > supertrend[i-1] if i > period else close[i] > lower_band[i]:
            # Bullish
            if lower_band[i] < supertrend[i-1] if i > period else True:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1] if i > period else 1
        else:
            # Bearish
            if upper_band[i] > supertrend[i-1] if i > period else True:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1] if i > period else -1
    
    return supertrend, direction, atr

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean().values
    wma2 = close_s.ewm(span=period, adjust=False).mean().values
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    hma_1w = calculate_hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Daily indicators - all computed before loop (Rule 8)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Supertrend(10, 3)
    supertrend, st_direction, atr = calculate_supertrend(high, low, close, 10, 3.0)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0)
    rsi = 100 - 100 / (1 + rs)
    
    # EMA(50) for additional trend filter
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # EMA(200) for major trend
    ema200 = close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume SMA(20) for volume confirmation
    vol_sma = pd.Series(volume).rolling(20, min_periods=20).mean().values
    vol_ratio = volume / vol_sma
    
    # KAMA(14) for adaptive trend
    def calculate_kama(close, period=14):
        n = len(close)
        kama = np.zeros(n)
        kama[0] = close[0]
        
        for i in range(1, n):
            change = abs(close[i] - close[max(0, i-period)])
            volatility = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
            er = change / volatility if volatility > 0 else 0
            sc = (er * (2/(period+1) - 2/(period+1)) + 2/(period+1)) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        
        return kama
    
    kama = calculate_kama(close, 14)
    
    signals = np.zeros(n)
    SIZE = 0.30
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start after 200-day EMA is ready
        # HTF trend: price vs 1w HMA (Rule 2 - use aligned array)
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filters
        daily_bullish = close[i] > ema50[i] and close[i] > ema200[i]
        daily_bearish = close[i] < ema50[i] and close[i] < ema200[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # RSI conditions (loosened for daily to ensure trades)
        rsi_ok_long = rsi[i] < 70  # not extremely overbought
        rsi_ok_short = rsi[i] > 30  # not extremely oversold
        
        # Volume confirmation (1.0x = average volume)
        vol_ok = vol_ratio[i] > 0.8  # at least 80% of average
        
        # KAMA slope for momentum
        kama_slope = kama[i] - kama[i-5] if i >= 5 else 0
        kama_bullish = kama_slope > 0
        kama_bearish = kama_slope < 0
        
        # Stoploss and trailing logic (Rule 6) - 2.5*ATR
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 2.5 * atr[i]
            initial_stop = entry_price - 2.5 * atr[i]
            stop_level = max(trail_stop, initial_stop)
            if close[i] < stop_level:
                signals[i] = 0.0
                position_side = 0
                continue
        
        if position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            initial_stop = entry_price + 2.5 * atr[i]
            stop_level = min(trail_stop, initial_stop)
            if close[i] > stop_level:
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # Long: HTF bullish + daily bullish + supertrend bullish
            # Loosened conditions: need 2 of 3 trend filters + RSI + volume
            trend_score_long = int(htf_bullish) + int(daily_bullish) + int(st_bullish) + int(kama_bullish)
            
            if trend_score_long >= 2 and rsi_ok_long and vol_ok:
                signals[i] = SIZE
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            
            # Short: HTF bearish + daily bearish + supertrend bearish
            trend_score_short = int(htf_bearish) + int(daily_bearish) + int(st_bearish) + int(kama_bearish)
            
            if trend_score_short >= 2 and rsi_ok_short and vol_ok:
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
        else:
            # Hold position - maintain signal
            signals[i] = signals[i-1]
    
    return signals