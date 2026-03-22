#!/usr/bin/env python3
"""
Experiment #033: 1d Primary + 1w HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Daily timeframe with weekly trend confirmation can capture major moves
while avoiding whipsaws. Combining Ehlers Fisher Transform (reversal detection)
with KAMA (adaptive trend following) and Bollinger squeeze (volatility breakout)
should work across bull/bear/range regimes.

Key components:
1. 1w KAMA(21) for MAJOR trend bias (only trade WITH weekly trend)
2. Fisher Transform(9) for reversal entries (crosses at extremes)
3. Bollinger Band squeeze detection (low vol before breakout)
4. ATR(14) trailing stoploss at 2.5x
5. ADX(14) regime filter (>25 = trend, <20 = range)
6. Discrete position sizing (0.25-0.30) to minimize fee churn

Why this should work on 1d:
- Fisher Transform catches reversals better than RSI in bear markets
- KAMA adapts to volatility (fast in trends, slow in ranges)
- Weekly trend filter prevents counter-trend trades
- BB squeeze identifies low-volatility breakouts
- Target: 15-30 trades/year (appropriate for 1d)

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_bb_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - fast in trends, slow in ranges
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s.diff(er_period))
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Median price
    median = (high_s + low_s) / 2
    
    # Normalize to -1 to +1 range
    highest = median.rolling(window=period, min_periods=period).max()
    lowest = median.rolling(window=period, min_periods=period).min()
    
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, np.nan)
    
    normalized = ((median - lowest) / range_hl - 0.5) * 1.9
    normalized = normalized.clip(-0.99, 0.99)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher_prev = fisher.shift(1)
    
    return fisher.values, fisher_prev.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and squeeze detection."""
    close_s = pd.Series(close)
    
    # Middle band (SMA)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    
    # Standard deviation
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    # Upper and lower bands
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    # Bandwidth (for squeeze detection)
    bandwidth = (upper - lower) / middle
    
    # Percent B (position within bands)
    percent_b = (close - lower) / (upper - lower)
    percent_b = np.nan_to_num(percent_b, nan=0.5)
    percent_b = np.clip(percent_b, 0, 1)
    
    return upper, lower, middle, bandwidth, percent_b

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    plus_dm_s = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / tr_s.replace(0, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    kama_1w_21 = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1w_21_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1d_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_prev = calculate_fisher_transform(high, low, 9)
    bb_upper, bb_lower, bb_middle, bb_bandwidth, bb_pct_b = calculate_bollinger_bands(close, 20, 2.0)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1w_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_pct_b[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        # === WEEKLY TREND BIAS (MAJOR) ===
        # Price above 1w KAMA = bullish bias (prefer longs)
        # Price below 1w KAMA = bearish bias (prefer shorts)
        trend_1w_bullish = close[i] > kama_1w_21_aligned[i]
        trend_1w_bearish = close[i] < kama_1w_21_aligned[i]
        
        # === DAILY TREND CONFIRMATION ===
        trend_1d_bullish = close[i] > kama_1d_21[i]
        trend_1d_bearish = close[i] < kama_1d_21[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 25 = trending market
        # ADX < 20 = ranging market
        is_trending = adx_14[i] > 25
        is_ranging = adx_14[i] < 20
        
        # === BOLLINGER SQUEEZE DETECTION ===
        # Low bandwidth = compression before breakout
        # Use 6-month percentile for bandwidth
        bb_squeeze = bb_bandwidth[i] < np.nanpercentile(bb_bandwidth[max(0,i-126):i+1], 30) if i >= 126 else bb_bandwidth[i] < 0.05
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_long_cross = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_cross = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow entries at extreme levels without cross
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in ranging market
        if is_ranging:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: Weekly bullish + Daily bullish + Fisher long signal
        # Secondary: BB squeeze breakout + Fisher oversold
        if trend_1w_bullish:
            # Trend-following long
            if trend_1d_bullish and (fisher_long_cross or fisher_oversold):
                new_signal = current_size
            # Pullback entry in uptrend
            elif trend_1d_bullish and bb_pct_b[i] < 0.3 and adx_14[i] > 15:
                new_signal = current_size * 0.8
        
        # SHORT ENTRIES
        # Primary: Weekly bearish + Daily bearish + Fisher short signal
        # Secondary: BB squeeze breakdown + Fisher overbought
        if trend_1w_bearish:
            # Trend-following short
            if trend_1d_bearish and (fisher_short_cross or fisher_overbought):
                new_signal = -current_size
            # Rally entry in downtrend
            elif trend_1d_bearish and bb_pct_b[i] > 0.7 and adx_14[i] > 15:
                new_signal = -current_size * 0.8
        
        # === RANGING MARKET MEAN REVERSION ===
        # Only if weekly trend is neutral (price near KAMA)
        if is_ranging:
            price_near_kama = abs(close[i] - kama_1w_21_aligned[i]) / kama_1w_21_aligned[i] < 0.03
            if price_near_kama:
                if fisher_oversold and bb_pct_b[i] < 0.2:
                    new_signal = current_size * 0.5
                elif fisher_overbought and bb_pct_b[i] > 0.8:
                    new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2 months on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.5
            elif trend_1w_bearish and fisher[i] > 1.0:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1w_bearish and fisher[i] > 1.5:
                trend_reversal = True
            if position_side < 0 and trend_1w_bullish and fisher[i] < -1.5:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals