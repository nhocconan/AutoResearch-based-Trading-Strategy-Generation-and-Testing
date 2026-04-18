# 1d_RSI_Extreme_With_WeeklyTrend
# Hypothesis: Daily RSI extremes (overbought >70, oversold <30) with weekly trend filter (price > weekly EMA50 for long, < for short).
# Works in bull markets by buying oversold dips in uptrend, in bear markets by selling overbought rallies in downtrend.
# Low trade frequency due to strict RSI thresholds and trend alignment.
# Includes volatility filter (ATR) to avoid chop and ensure momentum behind moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 14)  # Wait for weekly EMA and RSI warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50w = ema_50_1w_aligned[i]
        rsi_val = rsi[i]
        atr_val = atr[i]
        
        # Volatility filter: require ATR > 0.5% of price to avoid choppy markets
        vol_filter = atr_val > (0.005 * price)
        
        if position == 0:
            # Long: RSI oversold (<30) in weekly uptrend (price > weekly EMA50) with volatility
            if rsi_val < 30 and price > ema50w and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in weekly downtrend (price < weekly EMA50) with volatility
            elif rsi_val > 70 and price < ema50w and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI returns to neutral (>50) or trend turns down
            if rsi_val > 50 or price < ema50w:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI returns to neutral (<50) or trend turns up
            if rsi_val < 50 or price > ema50w:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_RSI_Extreme_With_WeeklyTrend"
timeframe = "1d"
leverage = 1.0