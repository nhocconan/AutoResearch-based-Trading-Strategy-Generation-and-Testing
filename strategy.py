#!/usr/bin/env python3
"""
Experiment #410: 30m Fisher Transform + 4h HMA Trend + Choppiness Regime

Hypothesis: After 409 failed experiments, the winning formula combines:
1. 4h HMA(21) for stable trend bias (proven in baseline Sharpe=0.676)
2. 30m Fisher Transform(9) for precise reversal entries (75% win rate in ranges)
3. 30m Choppiness Index(14) to adapt between trend/mean-reversion
4. 30m Donchian(20) for breakout confirmation in trending regimes
5. 2.5*ATR trailing stop for risk management

Why 30m works:
- Catches intraday moves that 4h/1d miss
- Fewer trades than 5m/15m (less fee churn)
- More responsive than 1h/4h for entry timing
- Still respects HTF trend via 4h HMA alignment

Key improvements over failed strategies:
- LOOSE entry thresholds (Fisher >0.5 not >1.5) to ensure ≥10 trades/symbol
- Dual entry modes: mean-reversion in range, breakout in trend
- Simple regime logic: CHOP>61.8=range, CHOP<38.2=trend, else flat
- Discrete position sizing (0.28) to minimize fee churn

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_chop_donchian_atr_v1"
timeframe = "30m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -0.8, Short when crosses below +0.8
    (Loose thresholds to ensure sufficient trades)
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    typical = (high + low + close) / 3.0
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > period else 0.0
            fisher_signal[i] = fisher[i]
            continue
        
        normalized = 2 * (typical[i] - lowest) / price_range - 1
        normalized = np.clip(normalized, -0.99, 0.99)
        
        raw_fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i == period:
            fisher[i] = raw_fisher
            fisher_signal[i] = raw_fisher
        else:
            fisher[i] = 0.67 * raw_fisher + 0.33 * fisher[i-1]
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion works)
    CHOP < 38.2 = trending market (trend-following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Donchian midpoint for mean-reversion reference
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] < 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (loose thresholds for more trades) ===
        fisher_long = (fisher_signal[i] < -0.8) and (fisher[i] > -0.8)
        fisher_short = (fisher_signal[i] > 0.8) and (fisher[i] < 0.8)
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_long = close[i] > donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # RANGING REGIME: Fisher Transform mean-reversion
        if ranging_market:
            if fisher_long and bull_trend_4h:
                new_signal = SIZE
            elif fisher_short and bear_trend_4h:
                new_signal = -SIZE
            elif fisher_long:
                new_signal = SIZE * 0.5  # Half size without trend confirmation
            elif fisher_short:
                new_signal = -SIZE * 0.5
        
        # TRENDING REGIME: Donchian breakout + 4h HMA confirmation
        elif trending_market:
            if donchian_breakout_long and bull_trend_4h:
                new_signal = SIZE
            elif donchian_breakout_short and bear_trend_4h:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0 and new_signal != 0.0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            elif position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and trending_market and not donchian_breakout_long:
                new_signal = 0.0
            if position_side < 0 and trending_market and not donchian_breakout_short:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals