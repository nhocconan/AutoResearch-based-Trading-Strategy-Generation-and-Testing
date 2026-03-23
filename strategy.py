#!/usr/bin/env python3
"""
Experiment #697: 1d Primary + 1w HTF — Vol Spike Mean Reversion + Regime Filter

Hypothesis: After 608 failed strategies, the clearest pattern is:
1. 1d timeframe consistently outperforms lower TFs (#693 Sharpe=0.105, #696 Sharpe=-0.071)
2. CHOP+CRSI combinations tried 50+ times with mostly negative results
3. Vol spike mean reversion is UNDERUTILIZED in our experiments
4. Research shows: "ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long. Captures vol crush after panic."

This strategy uses:
- ATR Ratio (7/30) for volatility spike detection (>2.0 = panic)
- Bollinger Bands (20, 2.5) for extreme oversold/overbought
- 1w HMA(21) for macro trend bias (only long above, only short below)
- Asymmetric sizing: 0.30 long, 0.20 short (crypto bias upward)
- Regime filter: ADX(14) < 25 = mean revert, ADX > 30 = trend follow

Why this might beat Sharpe=0.520:
- Vol spike reversion worked through 2022 crash (77% drop then rebound)
- BB 2.5 std dev catches true extremes (not just normal pullbacks)
- 1w HMA prevents counter-trend trades in strong trends
- Fewer but higher quality trades (target 25-40/year on 1d)
- Asymmetric sizing matches crypto's upward bias

Position sizing: 0.30 long / 0.20 short (discrete levels)
Target: 25-40 trades/year on 1d
Stoploss: 3.0*ATR trailing (wider for 1d noise)

CRITICAL: Entry conditions loosened to ensure >=10 trades/symbol
- ATR ratio > 1.8 (not 2.0) for more signals
- BB %B < 0.10 or > 0.90 (not just band touch)
- Either vol spike OR BB extreme = entry (not both required)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_volspike_bb_hma1w_regime_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with wider std dev for extremes."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # %B (position within bands)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    # Bandwidth for squeeze detection
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    return upper.values, lower.values, pct_b.values, bandwidth.values

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for vol spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    bb_upper, bb_lower, bb_pct_b, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # 1d SMA for additional filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, asymmetric for crypto bias)
    POSITION_SIZE_LONG = 0.30
    POSITION_SIZE_SHORT = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # ADX regime tracking with hysteresis
    prev_adx_regime = 0  # 0=neutral, 1=trend, 2=range
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(bb_pct_b[i]) or np.isnan(atr_ratio[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO TREND BIAS ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-5] if i >= 5 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-5] if i >= 5 else False
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ADX REGIME (with hysteresis) ===
        adx_val = adx_14[i]
        
        # Hysteresis: trend regime needs ADX>30 to enter, <22 to exit
        if adx_val > 30.0:
            adx_regime = 1  # Trending
        elif adx_val < 22.0:
            adx_regime = 2  # Range
        else:
            adx_regime = prev_adx_regime  # Keep previous regime
        
        prev_adx_regime = adx_regime
        is_trend_regime = (adx_regime == 1)
        is_range_regime = (adx_regime == 2)
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike_high = atr_ratio[i] > 1.8  # Vol spike (panic)
        vol_spike_low = atr_ratio[i] < 1.0   # Vol crush (calm)
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = bb_pct_b[i] < 0.10  # Near/below lower band
        bb_extreme_high = bb_pct_b[i] > 0.90  # Near/above upper band
        bb_below_lower = close[i] < bb_lower[i]
        bb_above_upper = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY (Mean Reversion after Vol Spike) ---
        # Condition 1: Vol spike + BB extreme low + not in strong downtrend
        if vol_spike_high and (bb_extreme_low or bb_below_lower):
            if price_above_hma_1w or (price_above_sma200 and not hma_1w_slope_bear):
                new_signal = POSITION_SIZE_LONG
        
        # Condition 2: Range regime + BB extreme (pure mean reversion)
        elif is_range_regime and bb_extreme_low:
            if not hma_1w_slope_bear:  # Not strongly bearish on 1w
                new_signal = POSITION_SIZE_LONG
        
        # Condition 3: Trend regime + pullback to support (trend follow)
        elif is_trend_regime and hma_1w_slope_bull and price_above_hma_1w:
            if bb_pct_b[i] < 0.30 and not vol_spike_high:  # Pullback, not panic
                new_signal = POSITION_SIZE_LONG
        
        # --- SHORT ENTRY (Mean Reversion after Vol Spike) ---
        # Condition 1: Vol spike + BB extreme high + not in strong uptrend
        if vol_spike_high and (bb_extreme_high or bb_above_upper):
            if price_below_hma_1w or (price_below_sma200 and not hma_1w_slope_bull):
                new_signal = -POSITION_SIZE_SHORT
        
        # Condition 2: Range regime + BB extreme (pure mean reversion)
        elif is_range_regime and bb_extreme_high:
            if not hma_1w_slope_bull:  # Not strongly bullish on 1w
                new_signal = -POSITION_SIZE_SHORT
        
        # Condition 3: Trend regime + pullback to resistance (trend follow)
        elif is_trend_regime and hma_1w_slope_bear and price_below_hma_1w:
            if bb_pct_b[i] > 0.70 and not vol_spike_high:  # Rally, not panic
                new_signal = -POSITION_SIZE_SHORT
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing for 1d noise) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON MACRO TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1w_slope_bear and price_below_hma_1w and adx_val > 25:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1w_slope_bull and price_above_hma_1w and adx_val > 25:
                new_signal = 0.0
        
        # === EXIT ON VOL CRUSH (take profit) ===
        if in_position and vol_spike_low:
            # Vol has normalized, take partial profit
            if position_side > 0 and bb_pct_b[i] > 0.50:
                new_signal = POSITION_SIZE_LONG / 2  # Reduce to half
            elif position_side < 0 and bb_pct_b[i] < 0.50:
                new_signal = -POSITION_SIZE_SHORT / 2  # Reduce to half
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals