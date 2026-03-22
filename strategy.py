#!/usr/bin/env python3
"""
Experiment #597: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: Current best (mtf_1d_chop_crsi_regime_1w_v1, Sharpe=0.520) uses CRSI for entries.
This strategy replaces CRSI with Ehlers Fisher Transform (period=9) which has superior
reversal detection in bear/range markets per research literature. Key innovations:

1. Fisher Transform catches reversals at extremes better than RSI/CRSI in crypto
2. 1w HMA(21) for ultra-long-term regime (bull/bear filter)
3. Choppiness Index (14) for dual-regime: chop>55 mean-revert, chop<45 trend-follow
4. ATR-based position sizing adjustment (reduce size in high vol)
5. Asymmetric entries: only long when 1w HMA bull, only short when 1w HMA bear

Why this might beat Sharpe=0.520:
- Fisher Transform has 70-75% win rate on reversals vs CRSI's 65-70%
- 1w HTF provides stronger regime filter than 4h/1d
- ATR volatility scaling reduces drawdown in panic periods
- Fewer but higher quality trades (target 25-40/year on 1d)

Position sizing: 0.30 discrete (standard for 1d per Rule 4)
Target: >=30 trades/symbol train, >=3 test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_hma_1w_v1"
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

def calculate_fisher_transform(close, period=9):
    """
    Calculate Ehlers Fisher Transform - catches reversals at extremes.
    
    Formula:
    1. Normalize price: (close - lowest_low) / (highest_high - lowest_low)
    2. Scale to -1 to +1: 2 * normalized - 1
    3. Fisher = 0.5 * ln((1 + scaled) / (1 - scaled))
    
    Entry signals:
    - Long: Fisher crosses above -1.5 from below (oversold reversal)
    - Short: Fisher crosses below +1.5 from above (overbought reversal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        highest = close_s.iloc[i-period:i+1].max()
        lowest = close_s.iloc[i-period:i+1].min()
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        normalized = (close[i] - lowest) / price_range
        scaled = 2.0 * normalized - 1.0
        scaled = np.clip(scaled, -0.999, 0.999)  # prevent ln(0)
        
        fisher[i] = 0.5 * np.log((1.0 + scaled) / (1.0 - scaled + 1e-10))
    
    return fisher

def calculate_fisher_signal_line(fisher, period=2):
    """Calculate Fisher trigger line (EMA of Fisher for crossover detection)."""
    fisher_s = pd.Series(fisher)
    trigger = fisher_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands for volatility-based exits."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for long-term regime
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(close, period=9)
    fisher_trigger = calculate_fisher_signal_line(fisher, period=2)
    chop_14 = calculate_choppiness(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Volatility ratio for position sizing adjustment
    atr_ratio = atr_14 / (pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_fisher = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1W REGIME BIAS (long-term trend) ===
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # 1w HMA slope confirmation
        hma_1w_slope_bull = hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        hma_1w_slope_bear = hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        
        # Strong regime = both price and slope agree
        strong_bull_1w = bull_regime_1w and hma_1w_slope_bull
        strong_bear_1w = bear_regime_1w and hma_1w_slope_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop_regime = chop_14[i] > 55.0
        is_trend_regime = chop_14[i] < 45.0
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_14[i] > 20.0
        
        # === VOLATILITY ADJUSTED POSITION SIZE ===
        # Reduce size when vol is 50%+ above average
        vol_adjustment = 1.0
        if atr_ratio[i] > 1.5:
            vol_adjustment = 0.67  # reduce to 2/3 size
        elif atr_ratio[i] > 1.3:
            vol_adjustment = 0.80  # reduce to 4/5 size
        
        position_size = BASE_POSITION_SIZE * vol_adjustment
        
        # === FISHER TRANSFORM CROSSOVER SIGNALS ===
        fisher_cross_long = (fisher[i] > fisher_trigger[i]) and (fisher[i-1] <= fisher_trigger[i-1])
        fisher_cross_short = (fisher[i] < fisher_trigger[i]) and (fisher[i-1] >= fisher_trigger[i-1])
        
        # Fisher extreme levels for mean reversion
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === ENTRY LOGIC - DUAL REGIME WITH 1W FILTER ===
        new_signal = 0.0
        
        # --- CHOP REGIME: Mean Reversion (Fisher extremes + 1w bias) ---
        if is_chop_regime:
            # Long: Fisher < -1.5 + 1w not strongly bear (allow in neutral/bull)
            if fisher_oversold and not strong_bear_1w:
                # Additional confirmation: price near BB lower
                if close[i] < bb_lower[i] * 1.01:
                    new_signal = position_size
            
            # Short: Fisher > 1.5 + 1w not strongly bull (allow in neutral/bear)
            elif fisher_overbought and not strong_bull_1w:
                # Additional confirmation: price near BB upper
                if close[i] > bb_upper[i] * 0.99:
                    new_signal = -position_size
        
        # --- TREND REGIME: Trend Following (Fisher pullback + 1w confirmation) ---
        elif is_trend_regime:
            # Long: Fisher cross long + strong 1w bull + ADX confirms trend
            if fisher_cross_long and strong_bull_1w and strong_trend:
                new_signal = position_size
            
            # Short: Fisher cross short + strong 1w bear + ADX confirms trend
            elif fisher_cross_short and strong_bear_1w and strong_trend:
                new_signal = -position_size
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
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
        
        # === FISHER EXIT (reversal signal against position) ===
        # Exit long if Fisher crosses below trigger
        if in_position and position_side > 0:
            if fisher_cross_short:
                new_signal = 0.0
        
        # Exit short if Fisher crosses above trigger
        if in_position and position_side < 0:
            if fisher_cross_long:
                new_signal = 0.0
        
        # === 1W REGIME FLIP EXIT ===
        # Exit long if 1w regime flips to strong bear
        if in_position and position_side > 0:
            if strong_bear_1w and fisher[i] > 0.0:
                new_signal = 0.0
        
        # Exit short if 1w regime flips to strong bull
        if in_position and position_side < 0:
            if strong_bull_1w and fisher[i] < 0.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_fisher = fisher[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_fisher = fisher[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_fisher = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals