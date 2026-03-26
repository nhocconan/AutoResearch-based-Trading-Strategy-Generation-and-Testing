#!/usr/bin/env python3
"""
Experiment #028: 4h Bollinger Band Bounce + TRIX + 1d Trend

HYPOTHESIS: Price Mean Reversion at Bollinger Band extremes is a high-probability 
setup on 4h. When price touches lower BB band AND TRIX turns positive (momentum 
shifting), it marks exhaustion. Combined with 1d HMA trend alignment (bull 
market only long, bear market only short), this captures reversals in both 
directions. Bounce trades have better win rates than breakout trades.

KEY INSIGHTS from DB:
- CRSI/TRIX momentum reversals at BB extremes work (ETH: test Sharpe 1.32)
- HMA+Donchian+Volume combos work (SOL: test Sharpe 1.38-1.46)
- BB squeeze/bounce is proven mean reversion pattern

TIMEFRAME: 4h primary
HTF: 1d for trend alignment (filter longs in bear, shorts in bull)
TARGET: 75-200 total trades over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_bounce_trix_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=14):
    """TRIX - Triple EMA Oscillator"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    trix = ema3.pct_change() * 100
    return trix.values

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_bb(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # TRIX(14) for momentum
    trix = calculate_trix(close, period=14)
    
    # TRIX signal (9-period of TRIX)
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Bollinger Bands (20, 2.0)
    bb_upper, bb_mid, bb_lower = calculate_bb(close, period=20, std_dev=2.0)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
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
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(trix[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        trend_bullish = price_above_1d_hma
        
        # === LOCAL INDICATORS ===
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        bb_mid_val = bb_mid[i]
        
        trix_val = trix[i]
        trix_sig = trix_signal[i]
        
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === TRIX MOMENTUM SHIFT ===
        # TRIX crosses above signal = momentum turning bullish
        trix_cross_up = (trix[i] > trix_sig[i]) and (trix[i-1] <= trix_signal[i-1] if i > 1 else True)
        # TRIX crosses below signal = momentum turning bearish
        trix_cross_down = (trix[i] < trix_sig[i]) and (trix[i-1] >= trix_signal[i-1] if i > 1 else True)
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio_val > 1.2
        
        # === BB TOUCH (price at extreme) ===
        # Long: price touches or is very close to lower band
        touch_lower = low[i] <= bb_low[i] * 1.002
        # Short: price touches or is very close to upper band
        touch_upper = high[i] >= bb_up[i] * 0.998
        
        # === RSI FILTER ===
        # Avoid entering when RSI is neutral
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        
        desired_signal = 0.0
        
        # === NEW ENTRY LOGIC ===
        if not in_position:
            # === LONG ENTRY: BB lower touch + TRIX turning up + RSI oversold + volume ===
            if touch_lower and trix_cross_up and rsi_oversold and vol_confirm:
                desired_signal = SIZE
            
            # === SHORT ENTRY: BB upper touch + TRIX turning down + RSI overbought + volume ===
            if touch_upper and trix_cross_down and rsi_overbought and vol_confirm:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: Price returned to BB mid ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            if close[i] >= bb_mid_val:
                tp_triggered = True
        
        if in_position and position_side < 0:
            if close[i] <= bb_mid_val:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals