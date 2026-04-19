# 6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2
# Strategy: Breakout of Camarilla R1/S1 levels with volume confirmation and ATR volatility filter
# Hypothesis: Camarilla pivot levels act as intraday support/resistance. Breakouts above R1 or below S1
# with above-average volume and sufficient volatility (ATR > 20-period mean) indicate institutional
# participation and trend continuation. Works in both bull/bear markets by following price action
# and volume confirmation rather than directional bias.
# Timeframe: 6h (balances signal frequency and noise reduction)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price arrays
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's OHLC (not current day to avoid look-ahead)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_open = np.roll(open_prices, 1)
    
    # First bar: use current values as fallback (no look-ahead)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_open[0] = open_prices[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    R4 = pivot + (range_hl * 1.1 / 2)
    S4 = pivot - (range_hl * 1.1 / 2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_sum -= volume[i - 20]
            vol_count -= 1
        if vol_count > 0:
            vol_ma[i] = vol_sum / vol_count
        else:
            vol_ma[i] = 0.0
    
    volume_filter = volume > 1.5 * vol_ma
    
    # ATR filter: current ATR > 20-period average ATR (volatility filter)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar: no previous close
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR using Wilder's smoothing (smoothed moving average)
    atr = np.zeros(n)
    if n >= 1:
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20  # 20-period smoothed
    
    # ATR 20-period average for comparison
    atr_ma = np.zeros(n)
    atr_sum = 0.0
    atr_count = 0
    for i in range(n):
        atr_sum += atr[i]
        atr_count += 1
        if atr_count >= 20:
            atr_sum -= atr[i - 20]
            atr_count -= 1
        if atr_count > 0:
            atr_ma[i] = atr_sum / atr_count
        else:
            atr_ma[i] = 0.0
    
    atr_filter = atr > atr_ma
    
    # Generate signals
    signals = np.zeros(n)
    
    # Start from index 20 to ensure indicators are ready
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(atr_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price closes above R1 with volume and volatility confirmation
        if (close[i] > R1[i] and 
            volume_filter[i] and 
            atr_filter[i]):
            signals[i] = 0.25
        
        # Short breakdown: price closes below S1 with volume and volatility confirmation
        elif (close[i] < S1[i] and 
              volume_filter[i] and 
              atr_filter[i]):
            signals[i] = -0.25
    
    return signals