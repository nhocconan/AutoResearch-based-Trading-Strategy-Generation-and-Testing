#!/usr/bin/env python3
"""
Experiment #023: 1d Camarilla + Volume + Choppiness

HYPOTHESIS: Camarilla pivot levels (R3/S3) act as strong support/resistance
where institutional activity clusters. Combined with Choppiness regime filter
to avoid choppy markets, this captures mean-reversion trades at natural 
inflection points with clear entry/exit levels.

WHY IT WORKS IN BULL AND BEAR:
- Bull: S3 touches provide low-risk long entries near support
- Bear: R3 touches provide short entries during rallies to resistance
- Camarilla is symmetric - same formula works in both directions
- Choppiness filter keeps us out of trending markets where mean-reversion fails

TARGET: 30-80 trades over 4 years (7-20/year). HARD MAX: 150.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_camarilla_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (Camarilla mean reversion works)
    CHOP < 38.2 = trending market (Camarilla may fail - avoid)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_camarilla(high, low, close):
    """
    Camarilla Pivot Levels
    Uses yesterday's OHLC to compute today's pivot levels.
    R3 = close + (high - low) * 1.1 / 4
    S3 = close - (high - low) * 1.1 / 4
    """
    n = len(close)
    r3 = np.zeros(n)
    s3 = np.zeros(n)
    
    for i in range(1, n):
        h_l = high[i-1] - low[i-1]
        c = close[i-1]
        r3[i] = c + h_l * 1.1 / 4
        s3[i] = c - h_l * 1.1 / 4
    
    return r3, s3

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA50 for trend direction
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    r3, s3 = calculate_camarilla(high, low, close)
    
    # Volume confirmation (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Pre-compute RSI outside loop
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 210  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10: