#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h HTF trend (HMA21) and 1d HTF regime filter (ADX<20 for range, >25 for trend)
# - 4h HMA21 determines trend direction (long when price>HMA, short when price<HMA)
# - 1d ADX(14) filters regime: only trade when ADX>25 (trending) or ADX<20 (range) to avoid chop
# - 1h RSI(2) for precise entry: long when RSI<10 in uptrend, short when RSI>90 in downtrend
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Fixed position size 0.20 to control drawdown and minimize fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in bull/bear: trend following in trending markets, mean reversion in ranging markets

name = "1h_4h_1d_hma_adx_rsi_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA(21) - Hull Moving Average
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean()
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean()
        wma_diff = 2 * wma_half - wma_full
        hma = pd.Series(wma_diff).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        # Smoothed TR, +DM, -DM
        tr_rma = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        plus_dm_rma = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_rma = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
        # Directional Indicators
        plus_di = 100 * plus_dm_rma / tr_rma
        minus_di = 100 * minus_dm_rma / tr_rma
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
        return adx.values
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 1h RSI(2)
    def calculate_rsi(arr, period=2):
        if len(arr) < period + 1:
            return np.full_like(arr, np.nan)
        delta = np.diff(arr)
        seed = delta[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        if down == 0:
            rs = np.inf
        else:
            rs = up / down
        rsi = np.zeros_like(arr)
        rsi[:period] = 100. - 100. / (1. + rs)
        for i in range(period, len(arr)):
            delta_val = delta[i-1]
            if delta_val > 0:
                up_val = delta_val
                down_val = 0.
            else:
                up_val = 0.
                down_val = -delta_val
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            if down == 0:
                rs = np.inf
            else:
                rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi
    
    rsi_1h = calculate_rsi(close, 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(rsi_1h[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 (trending) or ADX < 20 (range) to avoid chop
        adx_val = adx_1d_aligned[i]
        if adx_val >= 20 and adx_val <= 25:
            signals[i] = 0.0  # chop zone, no trade
            continue
        
        if position == 1:  # Long position
            # Exit conditions: RSI > 50 or price < 4h HMA (trend change)
            if rsi_1h[i] > 50 or close[i] < hma_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: RSI < 50 or price > 4h HMA (trend change)
            if rsi_1h[i] < 50 or close[i] > hma_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic based on regime
            if adx_val > 25:  # Trending regime - follow 4h HMA trend
                if close[i] > hma_4h_aligned[i] and rsi_1h[i] < 30:
                    # Long in uptrend on RSI pullback
                    position = 1
                    signals[i] = 0.20
                elif close[i] < hma_4h_aligned[i] and rsi_1h[i] > 70:
                    # Short in downtrend on RSI bounce
                    position = -1
                    signals[i] = -0.20
            else:  # Range regime (ADX < 20) - mean reversion at extremes
                if rsi_1h[i] < 10:
                    # Long on extreme oversold
                    position = 1
                    signals[i] = 0.20
                elif rsi_1h[i] > 90:
                    # Short on extreme overbought
                    position = -1
                    signals[i] = -0.20
    
    return signals