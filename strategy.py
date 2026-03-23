#!/usr/bin/env python3
"""
Experiment #073: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: Daily timeframe with weekly HMA macro bias, using Ehlers Fisher Transform for
reversal entries and Choppiness Index for regime detection, will generate 20-40 trades/year
with Sharpe > 0.486. This addresses the bear market problem (2022 crash, 2025 decline) by
only taking longs in bullish weekly trend and shorts in bearish weekly trend.

Key innovations:
1) Ehlers Fisher Transform (period=9): Long when Fisher crosses above -1.5, short when crosses below +1.5
   - Superior to RSI for catching reversals in bear markets
2) Weekly HMA(21) for macro bias: Only long when price > 1w HMA, only short when price < 1w HMA
   - Prevents counter-trend trades that destroyed capital in 2022
3) Choppiness Index regime: CHOP > 55 = range (mean revert), CHOP < 45 = trend (breakout)
4) ATR volatility scaling: Reduce position size when ATR(14)/ATR(50) > 2.0
5) Asymmetric entries: Long bias in bull regime, short bias in bear regime
6) 3.0*ATR trailing stoploss for trend, 2.0*ATR for mean reversion

Why this should work:
- 1d proven timeframe (fewer trades = less fee drag, 20-40/year target)
- Fisher Transform catches reversals better than RSI (research-backed)
- Weekly HMA prevents deadly counter-trend trades in bear markets
- Regime switch adapts to market conditions (trend vs chop)
- Conservative sizing (0.25-0.30) limits drawdown during crashes

Position size: 0.25-0.30 (discrete, vol-scaled)
Stoploss: 2.0-3.0*ATR trailing
Target: 20-40 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_hma_regime_1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    Signals: Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            X = 0.67 * ((close[i] - lowest) / price_range) - 0.33
            X = np.clip(X, -0.99, 0.99)  # Prevent log domain errors
            fisher[i] = 0.5 * np.log((1 + X) / (1 - X + 1e-10))
            
            # Trigger line (1-period lag of Fisher)
            if i > period:
                trigger[i] = fisher[i-1]
            else:
                trigger[i] = fisher[i]
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
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
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_TREND = 0.30
    POSITION_SIZE_MR = 0.25
    MAX_POSITION_SIZE = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_type = 'none'  # 'trend' or 'mr'
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(atr_50[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === WEEKLY MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY FILTER ===
        vol_ratio = atr_14[i] / (atr_50[i] + 1e-10)
        extreme_vol = vol_ratio > 2.0
        
        # Volatility-based position sizing
        vol_scale = 1.0
        if vol_ratio > 1.8:
            vol_scale = 0.7
        elif vol_ratio > 1.5:
            vol_scale = 0.85
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0
        is_trending = chop_value < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_signal = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5 or fisher[i-1] <= -1.5)
        fisher_short_signal = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5 or fisher[i-1] >= 1.5)
        
        # Fisher extreme reversals
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADAPTIVE REGIME ENTRY ===
        new_signal = 0.0
        new_entry_type = 'none'
        
        # --- TRENDING REGIME: Breakout + Weekly Bias ---
        if is_trending and not extreme_vol:
            # Long: Donchian breakout + weekly bullish + Fisher confirming
            if breakout_long and price_above_hma_1w:
                if fisher[i] > -1.0:  # Fisher not oversold
                    new_signal = POSITION_SIZE_TREND * vol_scale
                    new_entry_type = 'trend'
            
            # Short: Donchian breakdown + weekly bearish + Fisher confirming
            elif breakout_short and price_below_hma_1w:
                if fisher[i] < 1.0:  # Fisher not overbought
                    new_signal = -POSITION_SIZE_TREND * vol_scale
                    new_entry_type = 'trend'
        
        # --- RANGING REGIME: Fisher Mean Reversion + Weekly Bias ---
        elif is_ranging and not extreme_vol:
            # Long: Fisher oversold + weekly not strongly bearish
            if fisher_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE_MR * vol_scale
                new_entry_type = 'mr'
            
            # Short: Fisher overbought + weekly not strongly bullish
            elif fisher_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE_MR * vol_scale
                new_entry_type = 'mr'
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold long if RSI not overbought
            if position_side > 0 and rsi_14[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
                new_entry_type = entry_type
            # Hold short if RSI not oversold
            elif position_side < 0 and rsi_14[i] > 30.0:
                new_signal = signals[i-1] if i > 0 else 0.0
                new_entry_type = entry_type
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        stoploss_mult = 3.0 if entry_type == 'trend' else 2.0
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - stoploss_mult * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + stoploss_mult * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
            new_entry_type = 'none'
        
        # === EXIT ON WEEKLY TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1w:
                new_signal = 0.0
                new_entry_type = 'none'
        
        if in_position and position_side < 0:
            if price_above_hma_1w:
                new_signal = 0.0
                new_entry_type = 'none'
        
        # === CAP POSITION SIZE ===
        if new_signal != 0.0:
            if abs(new_signal) > MAX_POSITION_SIZE:
                new_signal = np.sign(new_signal) * MAX_POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_type = new_entry_type
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_type = new_entry_type
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_type = 'none'
        
        signals[i] = new_signal
    
    return signals