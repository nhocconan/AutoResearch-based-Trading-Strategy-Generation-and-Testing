#/usr/bin/env python3
# 1d_WeeklyPivot_PriceAction_Reversal
# Hypothesis: Weekly pivot points act as institutional support/resistance. Price reversals from weekly S1/R1 with volume confirmation and H4 trend filter capture mean-reversion moves in ranging markets and pullbacks in trending markets, effective in both bull and bear regimes.
# Entry: Long when price touches weekly S1 with bullish rejection (close > open) + volume spike + H4 close above EMA50; Short when price touches weekly R1 with bearish rejection (close < open) + volume spike + H4 close below EMA50.
# Exit: Mean reversion to weekly pivot (mean reversion to equilibrium).
# Target: 10-25 trades/year on 1d to stay within optimal range while capturing institutional level reactions.

name = "1d_WeeklyPivot_PriceAction_Reversal"
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
    open_price = prices['open'].values
    volume = prices['volume'].values

    # === WEEKLY PIVOT CALCULATION (using weekly OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Weekly pivot points: P = (H+L+C)/3, S1 = 2*P - H, R1 = 2*P - L
    wp = (wh + wl + wc) / 3.0
    ws1 = 2.0 * wp - wh  # Weekly Support 1
    wr1 = 2.0 * wp - wl  # Weekly Resistance 1
    
    # Align weekly pivot levels to daily timeframe
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    ws1_aligned = align_htf_to_ltf(prices, df_1w, ws1)
    wr1_aligned = align_htf_to_ltf(prices, df_1w, wr1)

    # === H4 TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # H4 EMA50 for trend filter
    h4_close = df_4h['close'].values
    ema50_4h = pd.Series(h4_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # === VOLUME CONFIRMATION ===
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Warmup for EMA50 and rolling
        # Skip if any required value is NaN
        if (np.isnan(wp_aligned[i]) or np.isnan(ws1_aligned[i]) or 
            np.isnan(wr1_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        price_touch_support = abs(low[i] - ws1_aligned[i]) <= (0.001 * ws1_aligned[i])  # Within 0.1% of S1
        price_touch_resistance = abs(high[i] - wr1_aligned[i]) <= (0.001 * wr1_aligned[i])  # Within 0.1% of R1
        
        bullish_rejection = close[i] > open_price[i]  # Bullish candle
        bearish_rejection = close[i] < open_price[i]  # Bearish candle
        
        volume_spike = volume[i] > vol_avg_20[i] * 1.5
        
        h4_uptrend = close[i] > ema50_4h_aligned[i]
        h4_downtrend = close[i] < ema50_4h_aligned[i]

        if position == 0:
            # LONG: Price touches weekly S1 with bullish rejection + volume spike + H4 uptrend
            if (price_touch_support and bullish_rejection and 
                volume_spike and h4_uptrend):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches weekly R1 with bearish rejection + volume spike + H4 downtrend
            elif (price_touch_resistance and bearish_rejection and 
                  volume_spike and h4_downtrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to weekly pivot
            if close[i] >= wp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to weekly pivot
            if close[i] <= wp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals