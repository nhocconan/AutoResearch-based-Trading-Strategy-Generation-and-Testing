# 1d_KAMA_Trend_RSI_Pullback
# Hypothesis: On daily timeframe, KAMA identifies the long-term trend direction, RSI identifies pullback entries in the trend direction.
# Long when KAMA is rising and RSI pulls back from overbought in an uptrend (RSI < 40 after being >60).
# Short when KAMA is falling and RSI pulls back from oversold in a downtrend (RSI > 60 after being <40).
# Uses 1-week trend filter to avoid counter-trend trades in strong trends.
# Designed for low trade frequency (10-30 trades/year) to minimize fee drag on 1d timeframe.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).

name = "1d_KAMA_Trend_RSI_Pullback"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average) for trend
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # This is incorrect, fixing below
    
    # Correct calculation of volatility (sum of absolute changes over period)
    volatility_sum = np.zeros_like(change)
    for i in range(len(change)):
        if i < 10:
            volatility_sum[i] = np.sum(change[:i+1])
        else:
            volatility_sum[i] = np.sum(change[i-9:i+1])
    
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    # Handle first element
    kama_rising[0] = False
    kama_falling[0] = False

    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI conditions for pullback
    rsi_overbought = rsi > 60
    rsi_oversold = rsi < 40
    rsi_pullback_long = rsi_overbought & (rsi < 40)  # This will never be true, fixing logic
    rsi_pullback_short = rsi_oversold & (rsi > 60)   # This will never be true
    
    # Correct RSI pullback logic: looking for RSI to cross back from extreme
    rsi_above_60_prev = np.roll(rsi_overbought, 1)
    rsi_below_40_prev = np.roll(rsi_oversold, 1)
    rsi_above_60_prev[0] = False
    rsi_below_40_prev[0] = False
    
    # Long signal: RSI was above 60 and now below 40 (pullback from overbought)
    rsi_pullback_long = rsi_above_60_prev & (rsi < 40)
    # Short signal: RSI was below 40 and now above 60 (pullback from oversold)
    rsi_pullback_short = rsi_below_40_prev & (rsi > 60)

    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly trend: price above/below EMA50
    weekly_uptrend = close > ema50_1w_aligned
    weekly_downtrend = close < ema50_1w_aligned

    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (uptrend) + RSI pullback from overbought + weekly uptrend + volume
            if kama_rising[i] and rsi_pullback_long[i] and weekly_uptrend[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (downtrend) + RSI pullback from oversold + weekly downtrend + volume
            elif kama_falling[i] and rsi_pullback_short[i] and weekly_downtrend[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI reaches overbought
            if not kama_rising[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI reaches oversold
            if not kama_falling[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals