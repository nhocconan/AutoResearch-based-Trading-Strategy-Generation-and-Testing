#5m_VWAP_RSI_Filtered
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1h data ONCE before loop for VWAP and trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Calculate 1h VWAP
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    typical_price_1h = (high_1h + low_1h + close_1h) / 3.0
    vwap_numerator = np.cumsum(typical_price_1h * volume_1h)
    vwap_denominator = np.cumsum(volume_1h)
    vwap_1h = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(typical_price_1h, np.nan), 
                        where=vwap_denominator!=0)
    vwap_1h_aligned = align_htf_to_ltf(prices, df_1h, vwap_1h)
    
    # Calculate 1h EMA20 for trend filter
    ema_20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    
    # Calculate RSI on 5m data
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = (1 - alpha) * avg_gain[i-1] + alpha * gain[i]
        avg_loss[i] = (1 - alpha) * avg_loss[i-1] + alpha * loss[i]
    
    rs = np.divide(avg_gain, avg_loss, 
                   out=np.full_like(avg_gain, np.nan), 
                   where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vwap_val = vwap_1h_aligned[i]
        ema_20_val = ema_20_1h_aligned[i]
        rsi_val = rsi[i]
        
        # Skip if any value is NaN
        if (np.isnan(vwap_val) or np.isnan(ema_20_val) or 
            np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above VWAP and EMA20, RSI oversold
            if close_val > vwap_val and close_val > ema_20_val and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP and EMA20, RSI overbought
            elif close_val < vwap_val and close_val < ema_20_val and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below VWAP or RSI overbought
            if close_val < vwap_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above VWAP or RSI oversold
            if close_val > vwap_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# VWAP + RSI strategy on 5m timeframe
# Uses 1h VWAP and EMA20 as trend filter
# Enters long when 5m price > 1h VWAP & EMA20 and RSI < 30
# Enters short when 5m price < 1h VWAP & EMA20 and RSI > 70
# Exits when price crosses VWAP or RSI reaches extreme
# VWAP provides dynamic support/resistance, RSI prevents chasing
# Target: 100-200 trades/year (5m frequency with filters)
name = "5m_VWAP_RSI_Filtered"
timeframe = "5m"
leverage = 1.0