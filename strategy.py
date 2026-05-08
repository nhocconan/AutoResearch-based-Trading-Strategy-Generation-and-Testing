# 1D_WeeklyTrend_With_Volume_Confirmation
# Hypothesis: 1-day trend following with weekly trend filter and volume confirmation
# Long when daily close > daily EMA(50) + weekly EMA(20) uptrend + volume spike
# Short when daily close < daily EMA(50) + weekly EMA(20) downtrend + volume spike
# Uses weekly trend as filter to avoid counter-trend trades in major trends
# Volume spike confirms institutional participation
# Targets 30-100 total trades over 4 years (7-25/year) to avoid fee drag

name = "1D_WeeklyTrend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily EMA(50) for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(ema50[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend = ema20_1w_aligned[i]
        daily_ema50 = ema50[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: daily close > daily EMA50 + weekly uptrend + volume spike
            if close[i] > daily_ema50 and close[i] > weekly_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: daily close < daily EMA50 + weekly downtrend + volume spike
            elif close[i] < daily_ema50 and close[i] < weekly_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: daily close < daily EMA50 OR weekly trend turns down
            if close[i] < daily_ema50 or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: daily close > daily EMA50 OR weekly trend turns up
            if close[i] > daily_ema50 or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals