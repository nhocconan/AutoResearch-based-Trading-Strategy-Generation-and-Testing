#!/usr/bin/env python3
"""
EXPERIMENT #107 - HMA_ADX_RSI_CHANDLIER_VOLREGIME_1H_V1
==================================================================================================
Hypothesis: Simplify the ensemble approach that worked in #096/#097. Use 1h timeframe for
better signal stability than 15m. Combine HMA trend + ADX strength + RSI pullback entries.
Key improvements over #106:
1. Single timeframe (1h) - less complexity, fewer resampling bugs
2. HMA(16/48) - faster trend detection than KAMA
3. ADX(14) > 25 filter - only trade strong trends
4. RSI(14) pullback entries - buy dips in uptrend, sell rallies in downtrend
5. Chandelier exit (3*ATR(22)) - proper trailing stop
6. BBW percentile for vol regime sizing - reduce size in high vol
7. Discrete signal levels (0.0, ±0.25, ±0.35) - minimize churn costs

Timeframe: 1h (optimal balance of signal quality vs trade frequency)
Expected: Sharpe > 5.0, DD < -20%, Trades > 50
"""

import numpy as np
import pandas as pd

name = "hma_adx_rsi_chandelier_volregime_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """ATR with Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, short_period=16, long_period=48):
    """Hull Moving Average - faster response than EMA"""
    close_s = pd.Series(close)
    
    wma_short = close_s.ewm(span=short_period, min_periods=short_period, adjust=False).mean()
    wma_long = close_s.ewm(span=long_period, min_periods=long_period, adjust=False).mean()
    
    wma_diff = 2.0 * wma_short - wma_long
    hma = wma_diff.ewm(span=int(np.sqrt(long_period)), min_periods=int(np.sqrt(long_period)), adjust=False).mean()
    
    return hma.fillna(0).values.copy()


def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    plus_sum = np.zeros(n)
    minus_sum = np.zeros(n)
    tr_sum = np.zeros(n)
    
    for i in range(period, n):
        if i == period:
            plus_sum[i] = np.sum(plus_dm[:period+1])
            minus_sum[i] = np.sum(minus_dm[:period+1])
            tr_sum[i] = np.sum(tr[:period+1])
        else:
            plus_sum[i] = plus_sum[i-1] - plus_sum[i-1]/period + plus_dm[i]
            minus_sum[i] = minus_sum[i-1] - minus_sum[i-1]/period + minus_dm[i]
            tr_sum[i] = tr_sum[i-1] - tr_sum[i-1]/period + tr[i]
        
        if tr_sum[i] > 0:
            plus_di[i] = 100 * plus_sum[i] / tr_sum[i]
            minus_di[i] = 100 * minus_sum[i] / tr_sum[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period] = np.mean(dx[:period+1])
    for i in range(period+1, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def calculate_rsi(close, period=14):
    """RSI with proper smoothing"""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    
    avg_gain = gain_s.ewm(span=period, min_periods=period).mean()
    avg_loss = loss_s.ewm(span=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).values.copy()


def calculate_bbw(close, period=20, std_dev=2.0):
    """Bollinger Band Width for volatility regime"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    
    return bandwidth.fillna(0).values.copy()


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    signals = np.zeros(n)
    
    # ===== Constants =====
    CHANDELIER_MULT = 3.0
    ATR_PERIOD = 22
    SIZE_LOW_VOL = 0.35
    SIZE_HIGH_VOL = 0.20
    VOL_THRESHOLD_PERCENTILE = 0.6
    ATR_TARGET_PCT = 0.015
    ADX_MIN = 25
    RSI_LONG_ENTRY = 45
    RSI_LONG_EXIT = 65
    RSI_SHORT_ENTRY = 55
    RSI_SHORT_EXIT = 35
    FIRST_VALID = 100
    
    # ===== Calculate indicators =====
    atr = calculate_atr(high, low, close, period=ATR_PERIOD)
    hma_short = calculate_hma(close, short_period=16, long_period=48)
    hma_long = calculate_hma(close, short_period=32, long_period=96)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bbw = calculate_bbw(close, period=20, std_dev=2.0)
    
    # ===== BBW percentile for vol regime =====
    bbw_percentile = np.zeros(n)
    valid_bbw = bbw[FIRST_VALID:][bbw[FIRST_VALID:] > 0]
    if len(valid_bbw) > 0:
        bbw_sorted = np.sort(valid_bbw)
        for i in range(FIRST_VALID, n):
            if bbw[i] > 0:
                bbw_percentile[i] = np.searchsorted(bbw_sorted, bbw[i]) / len(bbw_sorted)
    
    # ===== State variables =====
    prev_signal = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    tp_triggered = False
    chandelier_stop = 0.0
    
    for i in range(FIRST_VALID, n):
        # Skip invalid data
        if atr[i] == 0 or np.isnan(atr[i]) or close[i] == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # ===== Trend detection =====
        hma_trend = 0
        if close[i] > hma_short[i] and hma_short[i] > hma_long[i]:
            hma_trend = 1
        elif close[i] < hma_short[i] and hma_short[i] < hma_long[i]:
            hma_trend = -1
        
        # ===== ADX trend strength =====
        adx_strong = adx[i] > ADX_MIN
        di_direction = 1 if plus_di[i] > minus_di[i] else -1
        
        # ===== RSI pullback detection =====
        rsi_signal = 0
        if hma_trend == 1 and rsi[i] >= RSI_LONG_ENTRY and rsi[i] < RSI_LONG_EXIT:
            rsi_signal = 1
        elif hma_trend == -1 and rsi[i] <= RSI_SHORT_ENTRY and rsi[i] > RSI_SHORT_EXIT:
            rsi_signal = -1
        
        # ===== Volatility regime =====
        is_low_vol = bbw_percentile[i] < VOL_THRESHOLD_PERCENTILE
        
        # ===== Chandelier Exit management =====
        if prev_signal != 0.0 and entry_price > 0:
            atr_stop = atr[i]
            
            if prev_signal > 0:  # Long position
                highest_high = max(highest_high, high[i])
                chandelier_stop = highest_high - CHANDELIER_MULT * atr_stop
                
                # Take profit at 2R - reduce to half
                if not tp_triggered and close[i] >= entry_price + 2 * CHANDELIER_MULT * entry_atr:
                    signals[i] = prev_signal * 0.5
                    tp_triggered = True
                    chandelier_stop = max(chandelier_stop, entry_price + CHANDELIER_MULT * entry_atr)
                    prev_signal = signals[i]
                    continue
                
                # Stop loss - Chandelier exit
                if close[i] < chandelier_stop:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_high = 0.0
                    continue
                    
            else:  # Short position
                lowest_low = min(lowest_low, low[i])
                chandelier_stop = lowest_low + CHANDELIER_MULT * atr_stop
                
                # Take profit at 2R - reduce to half
                if not tp_triggered and close[i] <= entry_price - 2 * CHANDELIER_MULT * entry_atr:
                    signals[i] = prev_signal * 0.5
                    tp_triggered = True
                    chandelier_stop = min(chandelier_stop, entry_price - CHANDELIER_MULT * entry_atr)
                    prev_signal = signals[i]
                    continue
                
                # Stop loss - Chandelier exit
                if close[i] > chandelier_stop:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    tp_triggered = False
                    lowest_low = 0.0
                    continue
        
        # ===== Generate signal =====
        if prev_signal != 0.0:
            # Hold position if trend agrees, exit if trend flips or ADX weakens
            if hma_trend == 0 or hma_trend != np.sign(prev_signal) or not adx_strong:
                signals[i] = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
                tp_triggered = False
            else:
                signals[i] = prev_signal
        else:
            # New entry requires: trend + ADX strength + RSI pullback
            if hma_trend != 0 and adx_strong and rsi_signal != 0 and hma_trend == rsi_signal:
                # Volatility-adjusted position sizing
                atr_pct = atr[i] / close[i] if close[i] > 0 else 0
                vol_adj = min(1.3, max(0.7, ATR_TARGET_PCT / atr_pct)) if atr_pct > 0 else 1.0
                
                base_size = SIZE_LOW_VOL if is_low_vol else SIZE_HIGH_VOL
                position_size = np.clip(base_size * vol_adj, 0.15, SIZE_LOW_VOL)
                
                if hma_trend == 1:
                    signals[i] = position_size
                    entry_price = close[i]
                    entry_atr = atr[i]
                    highest_high = high[i]
                    chandelier_stop = highest_high - CHANDELIER_MULT * entry_atr
                    prev_signal = signals[i]
                    tp_triggered = False
                else:
                    signals[i] = -position_size
                    entry_price = close[i]
                    entry_atr = atr[i]
                    lowest_low = low[i]
                    chandelier_stop = lowest_low + CHANDELIER_MULT * entry_atr
                    prev_signal = signals[i]
                    tp_triggered = False
            else:
                signals[i] = 0.0
                prev_signal = 0.0
    
    # Clip to max position size
    signals = np.clip(signals, -0.40, 0.40)
    
    return signals