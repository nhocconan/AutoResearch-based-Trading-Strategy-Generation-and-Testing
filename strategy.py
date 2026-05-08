# 1h_RSI_Pullback_4hTrend_1dTrendFilter_Volume
# Hypothesis: 1h RSI pullback strategy with 4h EMA50 trend filter and 1d trend confirmation.
# Long when RSI < 40 and price > 4h EMA50 and 1d EMA50 > previous day's EMA50.
# Short when RSI > 60 and price < 4h EMA50 and 1d EMA50 < previous day's EMA50.
# Volume filter: volume > 1.5x 20-period average to avoid low-volume noise.
# Uses 4h for trend direction, 1d for trend confirmation, 1h for entry timing (pullback).
# Targets 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_RSI_Pullback_4hTrend_1dTrendFilter_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data once for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend confirmation
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        ema50_4h_val = ema50_4h_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        rsi_val = rsi[i]
        vol_spike_val = vol_spike[i]
        
        # 1d trend: current EMA50 > previous EMA50 for uptrend, < for downtrend
        ema50_1d_prev = ema50_1d_aligned[i-1] if i > 0 else ema50_1d_val
        
        if position == 0:
            # Enter long: RSI oversold, price above 4h EMA50, 1d uptrend, volume spike
            if rsi_val < 40 and close_val > ema50_4h_val and ema50_1d_val > ema50_1d_prev and vol_spike_val:
                signals[i] = 0.20
                position = 1
            # Enter short: RSI overbought, price below 4h EMA50, 1d downtrend, volume spike
            elif rsi_val > 60 and close_val < ema50_4h_val and ema50_1d_val < ema50_1d_prev and vol_spike_val:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or price below 4h EMA50
            if rsi_val > 60 or close_val < ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI oversold or price above 4h EMA50
            if rsi_val < 40 or close_val > ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals