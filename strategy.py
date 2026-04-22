# 1d_Trend_Squeeze_With_Pullback_v1
# Hypothesis: On 1d timeframe, use Bollinger Band squeeze (low volatility) to identify consolidation,
# then enter on breakout in direction of 200-day EMA with volume confirmation.
# Works in bull/bear markets: squeeze identifies breakout readiness, EMA200 filters trend direction.
# Low-frequency signals (target: 10-30 trades/year) reduce fee drag.
# Uses 1w EMA50 as higher-timeframe trend filter for robustness.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width below 20-period mean width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma  # Low volatility = consolidation
    
    # 200-day EMA for trend filter
    ema200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: 2x 20-period volume average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 2.0 * vol_ma20
    
    # Higher timeframe trend filter: 1w EMA50
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):  # Wait for 200 EMA warmup
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema200[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger breakout above upper band with volume,
            # price above EMA200 (uptrend), and 1w EMA50 rising
            if (close[i] > bb_upper[i] and vol_confirm[i] and 
                close[i] > ema200[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger breakout below lower band with volume,
            # price below EMA200 (downtrend), and 1w EMA50 falling
            elif (close[i] < bb_lower[i] and vol_confirm[i] and 
                  close[i] < ema200[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle Bollinger Band (mean reversion)
            if position == 1:
                if close[i] < bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Trend_Squeeze_With_Pullback_v1"
timeframe = "1d"
leverage = 1.0