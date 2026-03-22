#!/usr/bin/env python3
"""
Experiment #476: 30m Multi-Regime with 4h HMA Trend + Volume Confirmation

Hypothesis: 30m timeframe needs faster HTF reference (4h not 1w) and looser entry 
conditions to generate sufficient trades. Key innovations:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Bull: price > 4h HMA (favor long)
   - Bear: price < 4h HMA (favor short)
   - Faster than weekly, more responsive for 30m

2. CHOPPINESS INDEX (14) REGIME:
   - CHOP > 61.8 = range (mean-reversion)
   - CHOP < 38.2 = trend (breakout)
   
3. LOOSE RSI THRESHOLDS for trade generation:
   - Long: RSI < 35 (not 30, ensures more trades)
   - Short: RSI > 65 (not 70, ensures more trades)
   
4. VOLUME CONFIRMATION:
   - Volume > 1.3 * SMA(volume, 20) confirms breakout validity
   - Prevents false breakouts on low volume
   
5. ATR(14) STOPLOSS at 2.5x:
   - Tighter for 30m timeframe volatility
   - Signal → 0 when stopped

6. POSITION SIZING: 0.25 discrete
   - Conservative for 30m swings
   - Discrete levels minimize churn

Timeframe: 30m (REQUIRED)
HTF: 4h via mtf_data (call ONCE before loop)
Expected trades: 50-100/year per symbol (enough for Sharpe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_4h_hma_chop_rsi_vol_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
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
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (CRITICAL Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS INDEX REGIME ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.3 * vol_sma[i]
        
        # === ENTRY LOGIC - LOOSE THRESHOLDS FOR TRADES ===
        new_signal = 0.0
        
        # BULL REGIME: Favor long entries
        if bull_regime:
            if ranging_market:
                # Bull + Range: RSI mean-reversion long (loose threshold)
                if rsi[i] < 35:
                    new_signal = SIZE
            elif trending_market:
                # Bull + Trend: Momentum long with volume confirmation
                if close[i] > close[i-1] and vol_confirmed:
                    new_signal = SIZE
            else:
                # Neutral: RSI dip buy
                if rsi[i] < 40:
                    new_signal = SIZE
        
        # BEAR REGIME: Favor short entries
        if bear_regime:
            if ranging_market:
                # Bear + Range: RSI mean-reversion short (loose threshold)
                if rsi[i] > 65:
                    new_signal = -SIZE
            elif trending_market:
                # Bear + Trend: Momentum short with volume confirmation
                if close[i] < close[i-1] and vol_confirmed:
                    new_signal = -SIZE
            else:
                # Neutral: RSI rally sell
                if rsi[i] > 60:
                    new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
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