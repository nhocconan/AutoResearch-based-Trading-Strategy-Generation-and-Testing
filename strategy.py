#!/usr/bin/env python3
"""
Experiment #029: 1d Camarilla Pivot + Volume Spike + 1w SMA200 Trend

HYPOTHESIS: Camarilla pivot levels (S3/R3) from previous day mark institutional
support/resistance zones. Price bouncing from these levels with volume confirmation
captures high-probability mean-reversion trades. 1w SMA200 provides macro trend
filter to avoid countertrend trades. Choppiness Index keeps us out of ranging markets.

WHY 1d PRIMARY: Matches Binance 1d candle boundaries for Camarilla calculation.
Weekly HTF provides clean trend signal without noise.

TARGET: 50-100 total trades over 4 years (12-25/year).
Previous DB winner: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 
  achieved ETH test Sharpe=1.471 with 95 trades.

KEY DIFFERENCE: Using 1w SMA200 instead of local SMA200 for cleaner trend.
Entry: Price touches Camarilla level + volume spike + trend alignment.
Exit: ATR-based stoploss + RSI bounds + take profit at opposite level.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_camarilla_vol_1w_sma200_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_camarilla_pivots(high, low, close):
    """
    Calculate Camarilla pivot levels.
    S3/S4 = support zones, R3/R4 = resistance zones
    Previous day's HLC used for calculation.
    """
    n = len(close)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    h3 = np.full(n, np.nan)
    l3 = np.full(n, np.nan)
    h4 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    
    # Camarilla formulas (using previous day)
    for i in range(1, n):
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        range_hl = h - l
        
        h4[i] = c + (range_hl * 0.55)  # R4
        h3[i] = c + (range_hl * 0.275)  # R3
        l3[i] = c - (range_hl * 0.275)  # S3
        l4[i] = c - (range_hl * 0.55)  # S4
    
    return h3, l3, h4, l4

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (mean reversion favored)
    CHOP < 38.2 = trending market (momentum favored)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for macro trend direction
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    h3, l3, h4, l4 = calculate_camarilla_pivots(high, low, close)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for exit filter
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30  # Standard sizing for 1d strategy
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 220  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w SMA200) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range (mean reversion at pivots favored)
        # CHOP < 38.2 = trending (momentum favored - avoid pivot fades)
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === CAMARILLA LEVELS ===
        r3 = h3[i]
        s3 = l3[i]
        r4 = h4[i]
        s4 = l4[i]
        
        # Check if price is near Camarilla levels
        atr_local = atr_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === CHOPPY MARKET: Mean reversion at Camarilla levels ===
            if is_choppy:
                # Long: Price bounced from S3 with volume
                # Bounce condition: today's low touched near S3, now recovering
                if not np.isnan(s3) and s3 > 0:
                    touch_s3 = low[i] <= s3 + 0.5 * atr_local
                    bounce_s3 = close[i] > low[i] + 0.3 * (high[i] - low[i])  # Closed higher half
                    
                    if touch_s3 and bounce_s3 and price_above_1w_sma and vol_spike:
                        desired_signal = SIZE
                
                # Short: Price rejected from R3 with volume
                if not np.isnan(r3) and r3 > 0:
                    touch_r3 = high[i] >= r3 - 0.5 * atr_local
                    reject_r3 = close[i] < high[i] - 0.3 * (high[i] - low[i])  # Closed lower half
                    
                    if touch_r3 and reject_r3 and not price_above_1w_sma and vol_spike:
                        desired_signal = -SIZE
            
            # === TRENDING MARKET: Only trade with trend at outer levels ===
            if is_trending:
                # Long in uptrend: Price at S4 (deep support) with volume
                if not np.isnan(s4) and s4 > 0:
                    touch_s4 = low[i] <= s4 + 0.5 * atr_local
                    bounce_s4 = close[i] > low[i] + 0.4 * (high[i] - low[i])
                    
                    if touch_s4 and bounce_s4 and price_above_1w_sma and vol_spike:
                        desired_signal = SIZE
                
                # Short in downtrend: Price at R4 (deep resistance) with volume
                if not np.isnan(r4) and r4 > 0:
                    touch_r4 = high[i] >= r4 - 0.5 * atr_local
                    reject_r4 = close[i] < high[i] - 0.4 * (high[i] - low[i])
                    
                    if touch_r4 and reject_r4 and not price_above_1w_sma and vol_spike:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 3 bars) ===
        bars_held = i - entry_bar if in_position else 0
        min_hold_bars = 3
        
        if in_position and bars_held >= min_hold_bars:
            # Exit if RSI reaches extreme
            if position_side > 0 and rsi[i] > 70:
                desired_signal = 0.0
            if position_side < 0 and rsi[i] < 30:
                desired_signal = 0.0
        
        # === TAKE PROFIT (at 2.0 ATR or opposite level) ===
        if in_position and bars_held >= min_hold_bars:
            if position_side > 0:
                # Take profit if reached R3
                if not np.isnan(r3) and close[i] >= r3:
                    desired_signal = SIZE / 2  # Half position
                # Full exit at R4
                if not np.isnan(r4) and close[i] >= r4:
                    desired_signal = 0.0
            
            if position_side < 0:
                # Take profit if reached S3
                if not np.isnan(s3) and close[i] <= s3:
                    desired_signal = -SIZE / 2  # Half position
                # Full exit at S4
                if not np.isnan(s4) and close[i] <= s4:
                    desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
            # else: maintain position
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals