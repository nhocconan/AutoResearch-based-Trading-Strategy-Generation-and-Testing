# 12h strategy using daily POC (Point of Control) from volume profile with 1-week trend filter and volume confirmation
# POC acts as dynamic support/resistance in trending markets. Long when price pulls back to POC in uptrend, short when price bounces from POC in downtrend.
# Uses 1-week trend filter to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (12-37/year) to minimize fee drag and capture high-probability mean reversion within trend.
# Volume confirmation ensures institutional participation at POC level.

name = "12h_PullbackToPOC_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume profile POC calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily POC (Point of Control) - price level with highest volume
    poc = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        # Create volume profile for current day using 100 price bins between low and high
        if high_1d[i] <= low_1d[i] or volume_1d[i] <= 0:
            poc[i] = (high_1d[i] + low_1d[i]) / 2  # fallback to midpoint
            continue
            
        # Create 100 price bins
        bins = np.linspace(low_1d[i], high_1d[i], 101)
        # Assign each traded price to a bin (simplified: assume uniform distribution across range)
        # More realistic: use VWAP-like weighting, but for simplicity we'll use typical price weighted by volume
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        poc[i] = typical_price  # Simplified POC approximation
    
    # For better POC approximation, use volume-weighted close over recent period
    # Calculate VWAP over last 20 days as POC proxy
    typical_1d = (high_1d + low_1d + close_1d) / 3
    vwap_sum = np.zeros_like(close_1d)
    vol_sum = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        start_idx = max(0, i - 19)  # 20-day lookback
        vwap_sum[i] = np.sum(typical_1d[start_idx:i+1] * volume_1d[start_idx:i+1])
        vol_sum[i] = np.sum(volume_1d[start_idx:i+1])
        if vol_sum[i] > 0:
            poc[i] = vwap_sum[i] / vol_sum[i]
        else:
            poc[i] = typical_1d[i]
    
    # First few days need enough data
    for i in range(min(20, len(poc))):
        poc[i] = typical_1d[i]
    
    # Align POC to 12h timeframe
    poc_aligned = align_htf_to_ltf(prices, df_1d, poc)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(34) for trend filter (more stable)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = ema_34_1w[1:] > ema_34_1w[:-1]  # Rising weekly EMA
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])  # Align with daily index
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # 12x EMA(50) for intermediate trend and dynamic support/resistance
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(poc_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: pullback to POC in uptrend with volume confirmation
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > ema_50[i] and             # Above intermediate EMA
                abs((close[i] - poc_aligned[i]) / poc_aligned[i]) < 0.015 and  # Within 1.5% of POC
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: bounce from POC in downtrend with volume confirmation
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < ema_50[i] and            # Below intermediate EMA
                  abs((close[i] - poc_aligned[i]) / poc_aligned[i]) < 0.015 and  # Within 1.5% of POC
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves 2% away from POC or trend turns down
            if abs((close[i] - poc_aligned[i]) / poc_aligned[i]) > 0.02 or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves 2% away from POC or trend turns up
            if abs((close[i] - poc_aligned[i]) / poc_aligned[i]) > 0.02 or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals