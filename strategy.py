# 1h_4h1d_Momentum_Trend_Filter
# Hypothesis: 1h momentum with 4h trend filter and 1d volume confirmation.
# Long when: 1h RSI > 60, 4h EMA20 > EMA50 (uptrend), volume > 1.2x 20-bar average
# Short when: 1h RSI < 40, 4h EMA20 < EMA50 (downtrend), volume > 1.2x 20-bar average
# Uses 1h for entry timing, 4h for trend direction, 1d for volume filter.
# Designed for 15-35 trades/year to avoid fee drag. Works in bull/bear via trend filter.
name = "1h_4h1d_Momentum_Trend_Filter"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 4h EMA20 and EMA50 for trend
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d 20-period volume average for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema20_val = ema20_4h_aligned[i]
        ema50_val = ema50_4h_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: bullish momentum + uptrend + volume confirmation
            if rsi_val > 60 and ema20_val > ema50_val and vol > 1.2 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short entry: bearish momentum + downtrend + volume confirmation
            elif rsi_val < 40 and ema20_val < ema50_val and vol > 1.2 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: momentum fading or trend change
            if rsi_val < 50 or ema20_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: momentum fading or trend change
            if rsi_val > 50 or ema20_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals