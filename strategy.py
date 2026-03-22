#!/usr/bin/env python3
"""
Experiment #050: 30m Fisher Transform + 4h HMA Trend + Choppiness Regime
Hypothesis: 30m captures intermediate swings while 4h HMA filters major trend direction.
Fisher Transform excels at catching reversals in bear/range markets (proven in literature).
Choppiness Index distinguishes trending (CHOP<38.2) from ranging (CHOP>61.8) regimes.
Key insight: In trending regime, trade WITH 4h trend only. In ranging regime, mean-revert at extremes.
Position sizing: 0.25 base, 0.35 for strong confluence. Stoploss at 2.5*ATR trailing.
Why this might work: Fisher Transform has 70%+ win rate on reversals. 4h filter avoids counter-trend traps.
30m TF generates enough trades (target 50-100/year) while avoiding 5m/15m noise.
Must generate 10+ trades on train - entry conditions designed for adequate frequency.
Timeframe: 30m (REQUIRED for exp#050), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_chop_regime_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        if hh > ll:
            price_norm = (close[i] - ll) / (hh - ll)
        else:
            price_norm = 0.5
        
        # Constrain to 0.001-0.999 to avoid log(0)
        price_norm = np.clip(price_norm, 0.001, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + price_norm) / (1 - price_norm + 1e-10))
        
        # Smooth with EMA
        if i == period:
            fisher[i] = fisher_val
        else:
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    Range: 0-100
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                        abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / (sma + 1e-10)
    return upper, lower, bb_width

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, 20, 2.0)
    zscore = calculate_zscore(close, 20)
    chop = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # HMA on 30m for faster trend
    hma_30m = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation
        bull_trend_30m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_30m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Choppiness regime detection
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        transition_regime = 38.2 <= chop[i] <= 61.8
        
        # Fisher Transform signals
        fisher_cross_long = fisher_signal[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_short = fisher_signal[i] > 1.5 and fisher[i] <= 1.5
        
        # Fisher extreme levels (for mean reversion)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # RSI conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # Bollinger Band signals
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.01
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.99
        
        # Z-score filter
        zscore_extreme_long = zscore[i] < -1.5
        zscore_extreme_short = zscore[i] > 1.5
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.98
        price_near_ema21_short = close[i] >= ema_21[i] * 0.98 and close[i] <= ema_21[i] * 1.02
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) ===
        if trending_regime:
            # Long: Only with 4h trend + Fisher reversal signal
            if bull_trend_4h:
                if fisher_cross_long:
                    new_signal = SIZE_STRONG
                elif fisher_oversold and price_near_ema21_long and above_200:
                    new_signal = SIZE_BASE
                elif rsi_oversold and bull_trend_30m:
                    new_signal = SIZE_HALF
            
            # Short: Only with 4h trend + Fisher reversal signal
            elif bear_trend_4h:
                if fisher_cross_short:
                    new_signal = -SIZE_STRONG
                elif fisher_overbought and price_near_ema21_short and below_200:
                    new_signal = -SIZE_BASE
                elif rsi_overbought and bear_trend_30m:
                    new_signal = -SIZE_HALF
        
        # === RANGING REGIME (CHOP > 61.8) ===
        elif ranging_regime:
            # Long: Mean reversion at extremes (counter-trend OK in range)
            if fisher_oversold and price_at_lower_bb:
                if zscore_extreme_long or rsi_oversold:
                    new_signal = SIZE_BASE
            elif fisher_cross_long and zscore_extreme_long:
                new_signal = SIZE_HALF
            
            # Short: Mean reversion at extremes (counter-trend OK in range)
            if fisher_overbought and price_at_upper_bb:
                if zscore_extreme_short or rsi_overbought:
                    new_signal = -SIZE_BASE
            elif fisher_cross_short and zscore_extreme_short:
                new_signal = -SIZE_HALF
        
        # === TRANSITION REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: Only trade WITH 4h trend on strong signals
            if bull_trend_4h and fisher_cross_long and above_200:
                new_signal = SIZE_HALF
            elif bear_trend_4h and fisher_cross_short and below_200:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals