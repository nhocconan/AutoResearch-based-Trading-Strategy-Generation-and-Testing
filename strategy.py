# Investigate 1d_RSI_Reversal_With_Weekly_Trend_Filter
# Hypothesis: Weekly trend (1w EMA50) filters RSI(14) reversals on daily timeframe.
# Long: RSI < 30 and close > weekly EMA50 (uptrend). Short: RSI > 70 and close < weekly EMA50 (downtrend).
# Uses 25% position size to limit risk. Targets 10-25 trades/year to avoid fee drag.
# Works in bull/bear: Weekly trend filters counter-trend trades in strong trends, RSI captures mean reversion in ranges.

name = "1d_RSI_Reversal_With_Weekly_Trend_Filter"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Weekly trend filter: EMA(50) on close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        if position == 0:
            # LONG: RSI oversold (<30) and price above weekly EMA50 (uptrend filter)
            if rsi[i] < 30 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) and price below weekly EMA50 (downtrend filter)
            elif rsi[i] > 70 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend flip (price < weekly EMA50)
            if rsi[i] > 70 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend flip (price > weekly EMA50)
            if rsi[i] < 30 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals