#!/usr/bin/env python3
"""
Experiment #243: 1d Primary + 1w HTF — Dual Regime (Chop/Trend) + Connors RSI

Hypothesis: After 195+ failed experiments, research shows Choppiness Index + Connors RSI
works best on higher timeframes for BTC/ETH in bear/range markets. This strategy:

1. Choppiness Index (14) regime detection: CHOP>61.8=range, CHOP<38.2=trend
2. Range regime: Connors RSI mean reversion (CRSI<10 long, CRSI>90 short)
3. Trend regime: Donchian(20) breakout + 1d HMA(21) direction
4. 1w HMA(21) for macro bias alignment (only trade with weekly trend)
5. ATR(14) 2.5x trailing stoploss
6. Discrete sizing: 0.0, ±0.25, ±0.30

TARGET: 25-45 trades/year on 1d, Sharpe > 0.5 on ALL symbols
Key insight: Dual regime adapts to market conditions - mean revert in chop, trend follow otherwise.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_donchian_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    tr_s = pd.Series(tr)
    
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return np.nan_to_num(chop, nan=50.0)

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = calculate_rsi(streak, period=streak_period)
    
    # PercentRank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        percent_rank[i] = np.sum(window < close[i]) / rank_period * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    return np.nan_to_num(crsi, nan=50.0)

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # Calculate 1w HMA for macro trend (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        macro_bullish = price_above_hma_1w
        macro_bearish = price_below_hma_1w
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_14[i] > 61.8  # Range market
        trending_regime = chop_14[i] < 38.2  # Trend market
        neutral_regime = not choppy_regime and not trending_regime
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # --- RANGE REGIME: Connors RSI Mean Reversion ---
        if choppy_regime:
            # Long: CRSI < 10 (extreme oversold) + price above weekly HMA
            if crsi[i] < 15.0 and macro_bullish:
                desired_signal = POSITION_SIZE_FULL
            elif crsi[i] < 20.0 and not macro_bearish:
                desired_signal = POSITION_SIZE_HALF
            
            # Short: CRSI > 90 (extreme overbought) + price below weekly HMA
            elif crsi[i] > 85.0 and macro_bearish:
                desired_signal = -POSITION_SIZE_FULL
            elif crsi[i] > 80.0 and not macro_bullish:
                desired_signal = -POSITION_SIZE_HALF
        
        # --- TREND REGIME: Donchian Breakout + HMA Direction ---
        elif trending_regime:
            # Long breakout: price breaks Donchian upper + HMA bullish + weekly bullish
            breakout_long = close[i] > donchian_upper[i-1]  # break previous high
            hma_bullish = hma_21[i] > hma_21[i-5] if i >= 5 else False
            
            if breakout_long and hma_bullish and macro_bullish:
                desired_signal = POSITION_SIZE_FULL
            elif breakout_long and hma_bullish and not macro_bearish:
                desired_signal = POSITION_SIZE_HALF
            
            # Short breakout: price breaks Donchian lower + HMA bearish + weekly bearish
            breakout_short = close[i] < donchian_lower[i-1]  # break previous low
            hma_bearish = hma_21[i] < hma_21[i-5] if i >= 5 else False
            
            if breakout_short and hma_bearish and macro_bearish:
                desired_signal = -POSITION_SIZE_FULL
            elif breakout_short and hma_bearish and not macro_bullish:
                desired_signal = -POSITION_SIZE_HALF
        
        # --- NEUTRAL REGIME: Wait for clear signal ---
        elif neutral_regime:
            # Only enter on extreme CRSI in neutral regime
            if crsi[i] < 10.0 and macro_bullish:
                desired_signal = POSITION_SIZE_HALF
            elif crsi[i] > 90.0 and macro_bearish:
                desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit long if regime changes from range to strong trend down
        if in_position and position_side > 0:
            if trending_regime and hma_21[i] < hma_21[i-5] and macro_bearish:
                desired_signal = 0.0
        
        # Exit short if regime changes from range to strong trend up
        if in_position and position_side < 0:
            if trending_regime and hma_21[i] > hma_21[i-5] and macro_bullish:
                desired_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT (in range regime) ===
        if in_position and position_side > 0 and choppy_regime:
            if crsi[i] > 70.0:  # Take profit on mean reversion
                desired_signal = 0.0
        
        if in_position and position_side < 0 and choppy_regime:
            if crsi[i] < 30.0:  # Take profit on mean reversion
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals