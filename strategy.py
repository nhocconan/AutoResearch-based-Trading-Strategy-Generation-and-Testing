# 4H_MULTI_TF_BREAKOUT_COMBO  
# Hypothesis: Combine 4h Donchian breakout with 1d momentum filter and volume confirmation  
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.5x 20-period average  
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.5x 20-period average  
# Exit when price crosses back through Donchian middle (mean reversion) or momentum fails  
# Designed for low-frequency, high-conviction trades to avoid fee drag  
# Target: 20-50 total trades over 4 years (5-12/year)  

name = "4H_MULTI_TF_BREAKOUT_COMBO"  
timeframe = "4h"  
leverage = 1.0  

def generate_signals(prices):  
    n = len(prices)  
    if n < 50:  
        return np.zeros(n)  
    
    close = prices['close'].values  
    high = prices['high'].values  
    low = prices['low'].values  
    volume = prices['volume'].values  
    
    # Get 1d data for momentum filter  
    df_1d = get_htf_data(prices, '1d')  
    if len(df_1d) < 50:  
        return np.zeros(n)  
    
    # Donchian channels (20-period) on 4h  
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values  
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values  
    donchian_middle = (highest_high + lowest_low) / 2  
    
    # Volume filter: current volume > 1.5x 20-period average  
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  
    volume_filter = volume > (1.5 * vol_ma20)  
    
    # 1d EMA50 for momentum filter  
    close_1d = df_1d['close'].values  
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values  
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)  
    
    signals = np.zeros(n)  
    position = 0  # 0: flat, 1: long, -1: short  
    
    start_idx = 50  # Sufficient warmup  
    
    for i in range(start_idx, n):  
        # Skip if any critical data is NaN  
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or  
            np.isnan(volume_filter[i]) or np.isnan(ema50_1d_aligned[i])):  
            if position != 0:  
                signals[i] = 0.0  
                position = 0  
            continue  
        
        if position == 0:  
            # Long: breakout above Donchian high + momentum + volume  
            long_cond = (close[i] > highest_high[i]) and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]  
            # Short: breakout below Donchian low + momentum + volume  
            short_cond = (close[i] < lowest_low[i]) and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]  
            
            if long_cond:  
                signals[i] = 0.25  
                position = 1  
            elif short_cond:  
                signals[i] = -0.25  
                position = -1  
        elif position == 1:  
            # Long exit: price crosses below Donchian middle OR momentum fails  
            if close[i] < donchian_middle[i] or close[i] < ema50_1d_aligned[i]:  
                signals[i] = 0.0  
                position = 0  
            else:  
                signals[i] = 0.25  
        elif position == -1:  
            # Short exit: price crosses above Donchian middle OR momentum fails  
            if close[i] > donchian_middle[i] or close[i] > ema50_1d_aligned[i]:  
                signals[i] = 0.0  
                position = 0  
            else:  
                signals[i] = -0.25  
    
    return signals