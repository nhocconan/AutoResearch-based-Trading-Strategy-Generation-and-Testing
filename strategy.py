# 1d_RSI_Range_Bound_1wTrend_Filter
# Hypothesis: RSI mean reversion works better in ranging markets, but we can filter by 1w trend to avoid fighting strong trends.
# In 1d ranging markets (low ADX), go long when RSI < 30 and short when RSI > 70.
# In strong 1d trends (high ADX), follow the trend: long when RSI crosses above 50 in uptrend, short when below 50 in downtrend.
# Uses 1w EMA20 for trend filter to avoid whipsaws in major trend reversals.
# Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.

name = "1d_RSI_Range_Bound_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 1d RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # fill NaN with neutral 50
    
    # 1d ADX(14) for ranging vs trending detection
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Ranging market: ADX < 25
            if adx[i] < 25:
                # Mean reversion: RSI extremes
                if rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Trending market: ADX >= 25
                # Follow 1w trend with RSI crossing 50
                if ema20_1w_aligned[i] > ema20_1w_aligned[i-1]:  # 1w uptrend
                    if rsi[i] > 50 and rsi[i-1] <= 50:
                        signals[i] = 0.25
                        position = 1
                    else:
                        signals[i] = 0.0
                else:  # 1w downtrend
                    if rsi[i] < 50 and rsi[i-1] >= 50:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 60 (overbought) or 1w trend turns down
            if rsi[i] > 60 or ema20_1w_aligned[i] < ema20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 40 (oversold) or 1w trend turns up
            if rsi[i] < 40 or ema20_1w_aligned[i] > ema20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals