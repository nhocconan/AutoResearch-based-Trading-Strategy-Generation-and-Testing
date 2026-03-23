#!/usr/bin/env python3
"""
Experiment #213: 1d Primary + 1w HTF — Dual Regime (Choppiness + CRSI + HMA)

Hypothesis: Daily timeframe with weekly macro bias should produce 20-50 trades/year
with better risk-adjusted returns. Combines proven patterns:
1. Choppiness Index regime detection (CHOP>61.8 = range, CHOP<38.2 = trend)
2. Connors RSI for mean reversion entries in choppy markets (75% win rate)
3. HMA crossover for trend following in trending markets
4. 1w HMA for macro bias filter (only trade with weekly trend)

Key improvements from failed experiments:
- Simpler than #203's complex dual regime (that failed with Sharpe=-0.344)
- Uses 1d primary (proven higher TF works better than 4h/12h)
- CRSI instead of regular RSI for better mean reversion timing
- Discrete position sizing (0.25, 0.30) to minimize fee churn
- ATR trailing stoploss for risk management

TARGET: 20-50 trades/year, Sharpe > 0.50 on ALL symbols (BTC, ETH, SOL)
Position sizing: 0.0, ±0.25, ±0.30 (discrete levels)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_chop_hma_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100.0 * min(streak_abs[i], streak_period) / streak_period
        else:
            streak_rsi[i] = 100.0 * (1.0 - min(streak_abs[i], streak_period) / streak_period)
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_lower / (len(window) - 1) if len(window) > 1 else 50.0
    
    # CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1w HMA for macro trend (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market - mean reversion
        is_trending = chop[i] < 38.2  # Trending market - trend follow
        neutral_regime = not is_choppy and not is_trending
        
        # === TREND DETECTION (1d HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MEAN REVERSION (Choppy regime): CRSI extremes
        if is_choppy:
            # Long: CRSI < 10 (oversold) + price above weekly HMA
            if crsi[i] < 15.0 and price_above_hma_1w:
                new_signal = POSITION_SIZE_FULL
            # Short: CRSI > 85 (overbought) + price below weekly HMA
            elif crsi[i] > 85.0 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_FULL
            # Weaker signals in neutral weekly bias
            elif crsi[i] < 10.0:
                new_signal = POSITION_SIZE_HALF
            elif crsi[i] > 90.0:
                new_signal = -POSITION_SIZE_HALF
        
        # TREND FOLLOWING (Trending regime): HMA crossover + Donchian breakout
        elif is_trending:
            # Long: HMA bullish + price breaks Donchian upper + weekly bias
            if hma_bullish and close[i] > donchian_upper[i] * 0.995:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            # Short: HMA bearish + price breaks Donchian lower + weekly bias
            elif hma_bearish and close[i] < donchian_lower[i] * 1.005:
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        # NEUTRAL REGIME: Wait for clearer signals
        elif neutral_regime:
            # Only take strong CRSI extremes
            if crsi[i] < 8.0 and price_above_hma_1w:
                new_signal = POSITION_SIZE_HALF
            elif crsi[i] > 92.0 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime/trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought or trend still bullish
                if (is_choppy and crsi[i] < 75.0) or (is_trending and hma_bullish):
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold or trend still bearish
                if (is_choppy and crsi[i] > 25.0) or (is_trending and hma_bearish):
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if HMA crosses bearish (in trending regime)
        if in_position and position_side > 0 and is_trending and hma_bearish:
            new_signal = 0.0
        
        # Exit short if HMA crosses bullish (in trending regime)
        if in_position and position_side < 0 and is_trending and hma_bullish:
            new_signal = 0.0
        
        # Exit if weekly macro trend reverses against position
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            new_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit long when CRSI becomes overbought
        if in_position and position_side > 0 and crsi[i] > 80.0:
            new_signal = 0.0
        
        # Exit short when CRSI becomes oversold
        if in_position and position_side < 0 and crsi[i] < 20.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
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