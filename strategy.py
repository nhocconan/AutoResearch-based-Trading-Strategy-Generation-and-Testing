# 1d_KAMA_RSI_Chop_WeeklyTrend
# Hypothesis: Daily KAMA trend with weekly trend filter and RSI momentum
# Uses KAMA's adaptive nature to capture trends while avoiding whipsaws
# Weekly trend filter ensures alignment with higher timeframe momentum
# RSI provides entry timing on pullbacks within the trend
# Target: 15-25 trades per year (60-100 total over 4 years)

name = "1d_KAMA_RSI_Chop_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate KAMA on daily data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    volatility = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    volatility = np.concatenate([[np.nan]*9, volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14) on daily data
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price > KAMA (uptrend) + weekly uptrend + RSI > 50 + volume spike
            if (close[i] > kama_val and 
                close[i] > ema50_1w_val and 
                rsi_val > 50 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA (downtrend) + weekly downtrend + RSI < 50 + volume spike
            elif (close[i] < kama_val and 
                  close[i] < ema50_1w_val and 
                  rsi_val < 50 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA OR weekly trend turns down
            if (close[i] < kama_val or close[i] < ema50_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA OR weekly trend turns up
            if (close[i] > kama_val or close[i] > ema50_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals