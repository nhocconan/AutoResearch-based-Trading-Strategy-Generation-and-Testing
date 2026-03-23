#!/usr/bin/env python3
"""
Experiment #990: 1h Primary + 4h/12h HTF — Funding Rate Contrarian + Fisher Transform + Regime

Hypothesis: After 716 failed strategies, funding rate mean reversion is the MOST UNDERUTILIZED edge
for BTC/ETH (research shows Sharpe 0.8-1.5 through 2022 crash). Combined with Ehlers Fisher Transform
for precise entry timing and 4h HMA trend bias, this should work across ALL symbols.

Why 1h timeframe:
- Target 30-60 trades/year (manageable fee drag)
- Fisher Transform catches reversals faster than RSI
- Funding rate signals clearer on 1h than 4h
- Session filter (8-20 UTC) reduces low-liquidity trades

Key innovations:
1. FUNDING RATE Z-SCORE: Primary signal driver (contrarian edge)
   - Z > +1.5 → extreme longs → short signal
   - Z < -1.5 → extreme shorts → long signal
   - Works in BOTH bull and bear markets (unlike trend following)

2. EHLERS FISHER TRANSFORM: Entry timing precision
   - Period 9, long when Fisher crosses above -1.5 from below
   - Short when Fisher crosses below +1.5 from above
   - Catches reversals in bear rallies (critical for 2025 test period)

3. 4h HMA21: Trend bias filter (not entry trigger)
   - Only take longs when price > 4h HMA21
   - Only take shorts when price < 4h HMA21
   - Prevents counter-trend trades that whipsaw

4. CHOPPINESS INDEX: Regime-adaptive sizing
   - CHOP > 50 (range) → size = 0.20 (conservative)
   - CHOP < 45 (trend) → size = 0.30 (aggressive)
   - Reduces exposure during uncertain periods

5. SESSION FILTER: 8-20 UTC only
   - Avoids Asian session low liquidity
   - Reduces trade count by ~40%
   - Critical for meeting 30-60 trades/year target

6. ATR TRAILING STOP: 2.5x ATR(14)
   - Mandatory risk management
   - Signal → 0 when stop hit

Position sizing:
- BASE_SIZE = 0.30 (trending regime)
- REDUCED_SIZE = 0.20 (ranging regime)
- MAX magnitude = 0.35 (never 1.0!)
- DISCRETE levels only (0.0, ±0.20, ±0.30)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_funding_fisher_4h_hma_chop_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — highlights price reversals.
    
    Formula:
    1. Calculate typical price: (2*close + high + low) / 4
    2. Normalize to -1 to +1 range over lookback period
    3. Apply Fisher transform: 0.5 * ln((1+value)/(1-value))
    4. Smooth with EMA
    
    Entry signals:
    - Long: Fisher crosses above -1.5 from below
    - Short: Fisher crosses below +1.5 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, fisher_signal
    
    # Typical price
    typical = (2 * close + high + low) / 4.0
    
    for i in range(period, n):
        # Find highest and lowest over lookback
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            continue
        
        # Normalize to -1 to +1 range (with small epsilon to avoid div by zero)
        normalized = 2.0 * (typical[i] - lowest) / (highest - lowest + 1e-10) - 1.0
        normalized = np.clip(normalized, -0.999, 0.999)  # Avoid ln(0)
        
        # Fisher transform
        fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Smooth with simple EMA-like smoothing (previous value weighted)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_raw + 0.3 * fisher[i-1]
        else:
            fisher[i] = fisher_raw
    
    # Fisher signal line (1-period lag for crossover detection)
    fisher_signal[1:] = fisher[:-1]
    fisher_signal[0] = fisher[0] if not np.isnan(fisher[0]) else 0.0
    
    return fisher, fisher_signal

def calculate_funding_zscore(funding_series, period=30):
    """Z-score of funding rate over lookback period."""
    n = len(funding_series)
    zscore = np.full(n, np.nan)
    
    if n < period:
        return zscore
    
    for i in range(period - 1, n):
        window = funding_series[i-period+1:i+1]
        valid_window = window[~np.isnan(window)]
        
        if len(valid_window) < period // 2:
            continue
        
        mean = np.mean(valid_window)
        std = np.std(valid_window, ddof=1)
        
        if std > 1e-10:
            zscore[i] = (funding_series[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Load funding rate data if available (fallback to zeros if not)
    symbol = prices['symbol'].iloc[0] if 'symbol' in prices.columns else 'BTCUSDT'
    funding_path = f"data/processed/funding/{symbol}.parquet"
    try:
        df_funding = pd.read_parquet(funding_path)
        funding_rates = df_funding['funding_rate'].values
        # Align funding to prices length
        if len(funding_rates) >= n:
            funding_rates = funding_rates[-n:]
        else:
            funding_rates = np.concatenate([np.zeros(n - len(funding_rates)), funding_rates])
    except:
        funding_rates = np.zeros(n)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    fisher_1h, fisher_signal_1h = calculate_fisher_transform(high, low, close, period=9)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate funding z-score
    funding_z = calculate_funding_zscore(funding_rates, period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossover state
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(chop_1h[i]) or np.isnan(fisher_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === TREND BIAS (4h HTF HMA21) ===
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === MACRO CONFIRMATION (12h HTF HMA21) ===
        macro_bullish = close[i] > hma_12h_aligned[i]
        macro_bearish = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 50
        trending_regime = chop_1h[i] < 45
        
        # === FUNDING RATE CONTRARIAN (Primary Signal) ===
        funding_extreme_short = funding_z[i] > 1.5  # Too many longs → short
        funding_extreme_long = funding_z[i] < -1.5  # Too many shorts → long
        funding_moderate_short = funding_z[i] > 0.8
        funding_moderate_long = funding_z[i] < -0.8
        
        # === FISHER TRANSFORM CROSSOVER (Entry Timing) ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 100 and not np.isnan(fisher_1h[i]) and not np.isnan(fisher_signal_1h[i]):
            # Long: Fisher crosses above -1.5 from below
            if fisher_signal_1h[i] < -1.5 and fisher_1h[i] >= -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if fisher_signal_1h[i] > 1.5 and fisher_1h[i] <= 1.5:
                fisher_cross_short = True
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_1h[i] < 40
        rsi_overbought = rsi_1h[i] > 60
        rsi_neutral = 35 <= rsi_1h[i] <= 65
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        if in_session:
            # Primary: Funding extreme long + Fisher long cross + trend bullish
            if funding_extreme_long and fisher_cross_long and trend_bullish:
                desired_signal = BASE_SIZE
            # Secondary: Funding extreme long + Fisher long cross + macro bullish
            elif funding_extreme_long and fisher_cross_long and macro_bullish:
                desired_signal = REDUCED_SIZE
            # Tertiary: Funding moderate long + Fisher long cross + both trends bullish
            elif funding_moderate_long and fisher_cross_long and trend_bullish and macro_bullish:
                desired_signal = REDUCED_SIZE
            # Range regime: Funding extreme long + RSI oversold
            elif ranging_regime and funding_extreme_long and rsi_oversold:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        if in_session:
            # Primary: Funding extreme short + Fisher short cross + trend bearish
            if funding_extreme_short and fisher_cross_short and trend_bearish:
                desired_signal = -BASE_SIZE
            # Secondary: Funding extreme short + Fisher short cross + macro bearish
            elif funding_extreme_short and fisher_cross_short and macro_bearish:
                desired_signal = -REDUCED_SIZE
            # Tertiary: Funding moderate short + Fisher short cross + both trends bearish
            elif funding_moderate_short and fisher_cross_short and trend_bearish and macro_bearish:
                desired_signal = -REDUCED_SIZE
            # Range regime: Funding extreme short + RSI overbought
            elif ranging_regime and funding_extreme_short and rsi_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === ADJUST SIZE BY REGIME ===
        if desired_signal != 0.0:
            if ranging_regime:
                # Reduce size in ranging market
                if desired_signal > 0:
                    desired_signal = min(desired_signal, REDUCED_SIZE)
                else:
                    desired_signal = max(desired_signal, -REDUCED_SIZE)
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if trend_bullish and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if trend_bearish and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + RSI overbought
            if trend_bearish and rsi_1h[i] > 65:
                desired_signal = 0.0
            # Exit if funding flips extreme short
            if funding_extreme_short:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + RSI oversold
            if trend_bullish and rsi_1h[i] < 35:
                desired_signal = 0.0
            # Exit if funding flips extreme long
            if funding_extreme_long:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals