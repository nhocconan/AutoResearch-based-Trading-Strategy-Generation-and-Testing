# 1d_OBV_Signal_Trend_Filtered
# Hypothesis: Daily OBV trend with price above/below SMA200 filter. OBV confirms trend strength via volume-price relationship.
# Works in bull markets via breakouts and bear markets via trend following with volume confirmation.
# Target: <25 trades/year to minimize fee drift.
import numpy as np
import pandas as pd
from mdata import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily OBV for trend strength
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate OBV: cumulative volume * sign(price change)
    price_change = np.diff(close_1d, prepend=close_1d[0])
    obv = np.cumsum(volume_1d * np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0)))
    
    # Daily EMA200 for trend filter
    ema_1d_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 1d timeframe
    obv_aligned = align_htf_to_ltf(prices, df_1d, obv)
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Price relative to SMA200 (daily)
    sma_1d_200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma_1d_200_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if np.isnan(obv_aligned[i]) or np.isnan(ema_1d_200_aligned[i]) or np.isnan(sma_1d_200_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: OBV rising and price above SMA200
            if obv_aligned[i] > obv_aligned[i-1] and close[i] > sma_1d_200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: OBV falling and price below SMA200
            elif obv_aligned[i] < obv_aligned[i-1] and close[i] < sma_1d_200_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: OBV trend reversal or price crosses SMA200 in opposite direction
            if position == 1:
                if obv_aligned[i] < obv_aligned[i-1] or close[i] < sma_1d_200_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if obv_aligned[i] > obv_aligned[i-1] or close[i] > sma_1d_200_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_OBV_Signal_Trend_Filtered"
timeframe = "1d"
leverage = 1.0