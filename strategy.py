# 12h_Camarilla_Pivot_R1S1_Breakout_Volume_1dEMA34
# Hypothesis: On 12h timeframe, price breaking above daily Camarilla R1 or below S1 with volume > 1.5x daily average volume and price > 1d EMA34 (trend filter) captures institutional breakouts. 
# Works in bull/bear: EMA34 filter ensures we only trade with the daily trend, avoiding counter-trend whipsaws. Volume confirmation filters false breakouts.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).
name = "12h_Camarilla_Pivot_R1S1_Breakout_Volume_1dEMA34"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots, volume average, and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous day's values to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day will have NaN from roll, which is correct (no previous day)
    pivot_range = prev_high - prev_low
    r1 = prev_close + 1.1 * pivot_range / 12
    s1 = prev_close - 1.1 * pivot_range / 12
    
    # Calculate daily average volume (20-period) for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        ema34 = ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x daily average volume
        vol_confirm = vol > 1.5 * vol_ma
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = price > ema34
        price_below_ema = price < ema34
        
        if position == 0:
            # Long entry: price breaks above R1 + volume confirmation + price > EMA34
            if price > r1_val and vol_confirm and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + volume confirmation + price < EMA34
            elif price < s1_val and vol_confirm and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below R1 (or could use S1 as stop)
            if price < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above S1 (or could use R1 as stop)
            if price > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals