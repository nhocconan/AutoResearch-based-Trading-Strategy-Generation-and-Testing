# #!/usr/bin/env python3
"""
1d_Weekly_RSI2_MeanReversion
Hypothesis: Daily RSI(2) mean reversion with weekly trend filter and volume confirmation.
In bull markets, buy oversold dips in uptrend; in bear markets, sell overbought rallies in downtrend.
Uses weekly trend to avoid counter-trend trades, RSI(2) for short-term mean reversion,
volume spike to confirm institutional interest, and 1-day ATR stop to limit losses.
Target: 15-25 trades/year to minimize fee drag while capturing mean reversion edges.
"""

name = "1d_Weekly_RSI2_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get price, volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI(2) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI(2) calculation (2) and weekly EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend AND RSI(2) oversold (<10) with volume spike
            if close[i] > ema_34_1w_aligned[i] and rsi[i] < 10 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend AND RSI(2) overbought (>90) with volume spike
            elif close[i] < ema_34_1w_aligned[i] and rsi[i] > 90 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI(2) overbought (>70) OR weekly trend turns bearish
            if rsi[i] > 70 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI(2) oversold (<30) OR weekly trend turns bullish
            if rsi[i] < 30 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals